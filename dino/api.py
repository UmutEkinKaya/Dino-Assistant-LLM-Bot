"""HTTP katmanı. İnce tutulur: iş mantığı agent.py / factcheck.py'de, burada sadece
istek → fonksiyon → cevap ve hata → HTTP durum kodu çevirisi var.

Çalıştırma (proje klasöründe):   python -m uvicorn dino.api:app --reload
Test sayfası:                     http://127.0.0.1:8000/docs

Endpoint'ler 'async def' değil 'def': OpenAI istemcimiz senkron. FastAPI 'def' endpoint'leri
ayrı bir thread havuzunda çalıştırır, yani bir istek LLM'i beklerken diğerleri bloklanmaz.
(async'e geçmek için tüm zincirin AsyncOpenAI ile yazılması gerekir; el kitabı Bölüm 6.)
"""
import json
import logging
import queue
import re
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pathlib import Path

from . import agent, export, factcheck, llm
from .config import get_settings
from .retrieval import Retriever
from .schemas import (AskRequest, AskResponse, CheckRequest, CheckResponse, ExportFile, SourceInfo,
                      Ticket, TimelineStep, Usage)
from .tools import read_tickets

log = logging.getLogger("dino.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Açılışta bir kez: ayarları doğrula, indeksi ve embedding modelini yükle.
    s = get_settings()
    s.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(s.log_dir / "api.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    for name in ("httpx", "httpx2", "httpcore"):           # her LLM isteğini ayrıca loglamasın
        logging.getLogger(name).setLevel(logging.WARNING)
    app.state.retriever = Retriever.load()            # indeks yoksa servis AÇILMAZ (iyi bir şey)
    app.state.retriever.dense("warm up", 1)            # modeli ilk istekten önce ısıt
    log.info("hazır: %d chunk, model=%s", len(app.state.retriever.ids), s.llm_model)
    yield


app = FastAPI(title="Dino Bilgi Asistanı", version="1.0", lifespan=lifespan)


# ------------------------------------------------------------------------------------------
# Her isteğe bir kimlik + süre logu. Sorun olduğunda logda bu kimlikle ararsın.
# ------------------------------------------------------------------------------------------
@app.middleware("http")
async def request_log(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:8])
    t0 = time.time()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    log.info("%s %s %s %d %dms", rid, request.method, request.url.path, response.status_code,
             (time.time() - t0) * 1000)
    return response


# ------------------------------------------------------------------------------------------
# Hata çevirisi: LLM katmanının hataları → anlamlı HTTP cevapları (500 yerine)
# ------------------------------------------------------------------------------------------
@app.exception_handler(llm.LLMError)
async def llm_error(request: Request, exc: llm.LLMError):
    return JSONResponse(status_code=exc.status, content={"error": exc.code, "detail": str(exc)})


def _usage(m: dict, latency_ms: int) -> Usage:
    return Usage(calls=m["calls"], in_tok=m["in_tok"], out_tok=m["out_tok"],
                 usd=round(m["usd"], 6), latency_ms=latency_ms)


@app.get("/", include_in_schema=False)
def home():
    """Basit sohbet sayfası. Kendisi bir şey bilmez: tarayıcıdan /ask ve /check'i çağırır.
    Yani arayüz de API'nin sıradan bir kullanıcısı; aynı API'yi başka bir uygulama da kullanabilir."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "chunks": len(app.state.retriever.ids), "model": s.llm_model}


# ------------------------------------------------------------------------------------------
# Oturum hafızası: sohbetin doğrulanmış cevapları (dosyaya dökmek için). Bellekte, sınırlı:
# en çok 200 oturum x 20 tur; en eski oturum atılır. (Çok sunuculu kurulumda Redis gibi bir yer gerekir.)
# ------------------------------------------------------------------------------------------
SESSIONS: "OrderedDict[str, list[dict]]" = OrderedDict()
_sess_lock = threading.Lock()
MAX_SESSIONS, MAX_TURNS = 200, 20


def _session(sid: str) -> list[dict]:
    with _sess_lock:
        turns = SESSIONS.pop(sid, [])
        SESSIONS[sid] = turns                          # en son kullanılan sona
        while len(SESSIONS) > MAX_SESSIONS:
            SESSIONS.popitem(last=False)
        return turns


# ------------------------------------------------------------------------------------------
# Cevap önbelleği: aynı soru (geçmişsiz, normalize edilmiş) tekrar gelirse LLM'e gidilmez.
# Anahtarda model ve deney bayrakları da var: ayar değişince eski cevap dönmez. Bellekte, en çok 256.
# Dosya üreten cevaplar önbelleğe alınmaz (dosya her istekte yeniden üretilmeli).
# ------------------------------------------------------------------------------------------
ANSWERS: "OrderedDict[tuple, AskResponse]" = OrderedDict()
_ans_lock = threading.Lock()
MAX_ANSWERS = 256


def _cache_key(question: str) -> tuple:
    s = get_settings()
    q = re.sub(r"\s+", " ", question.lower()).strip().rstrip("?.!؟ ").replace("’", "'")
    return (q, s.llm_model, s.answer_schema_v2, s.live_tools, s.search_spillover)


def _remember(turns, question: str, answer, sources):
    if turns is not None:
        turns.append({"question": question, "answer": answer, "sources": sources})
        del turns[:-MAX_TURNS]


def run_ask(req: AskRequest, on_event=None) -> AskResponse:
    """/ask ve /ask/stream'in ortak işi. on_event: ilerleme adımlarını dışarı bildirir."""
    t0 = time.time()
    turns = _session(req.session_id) if req.session_id else None
    history = [t.model_dump() for t in req.history]
    with llm.meter() as m:
        # Takip sorusu önce netleştirilir; önbellek anahtarı bu net soru olur ("peki onu..." ≠ anahtar).
        standalone, pre = None, []
        if history:
            if on_event:
                on_event("Soru önceki konuşmaya göre netleştiriliyor")
            standalone = agent.condense(req.question, history)
            pre = [TimelineStep(t=0, text="Soru önceki konuşmaya göre netleştirildi")]
        shift = round(time.time() - t0, 2)
        key = _cache_key(standalone or req.question)
        with _ans_lock:
            hit = ANSWERS.get(key)
            if hit is not None:
                ANSWERS.move_to_end(key)
        if hit is not None:
            if on_event:
                on_event("⚡ Bu soru daha önce cevaplandı: önbellekten geliyor")
            out = hit.model_copy(deep=True)
            out.cached = True
            out.standalone_question = standalone if standalone and standalone != req.question else None
            out.usage = _usage(m, int((time.time() - t0) * 1000))
            out.timeline = pre + [TimelineStep(t=shift, text="Önbellekten (agent çalışmadı)")]
            if not req.debug:
                out.trace = None
            _remember(turns, standalone or req.question, out.answer, out.sources)
            return out
        r = agent.ask(req.question, app.state.retriever, history=history, allow_export=turns is not None,
                      on_event=on_event, condensed=standalone)
        r["latency_ms"] = int((time.time() - t0) * 1000)
        r["timeline"] = [x.model_dump() for x in pre] + [{"t": round(x["t"] + shift, 2), "text": x["text"]}
                                                        for x in r.get("timeline", [])]
    sources = _sources_meta(r["answer"], r.get("links"))
    if not r["only_export"]:                           # "bunu PDF yap" bir cevap değil, kaydedilmez
        _remember(turns, r.get("standalone_question") or req.question, r["answer"], sources)
    warnings = list(r["warnings"])
    files = [ExportFile(name=f["name"], url=f"/exports/{f['name']}", format=f["format"], turns=f["turns"],
                        kind="table") for f in r.get("files", [])]   # tablo aracı dosyayı kendisi üretti
    if r["export_request"] and turns is not None:
        # Model sadece "dosya istendi" dedi; dosyayı burada, doğrulanmış cevaplardan KOD üretir.
        x = r["export_request"]
        chosen = turns[-1:] if x["scope"] == "last_answer" else list(turns)
        if chosen:
            try:
                path = export.render(chosen, x["format"], x["title"])
                files.append(ExportFile(name=path.name, url=f"/exports/{path.name}", format=x["format"],
                                        turns=len(chosen)))
            except Exception as e:                      # dosya hatası cevabı düşürmesin
                log.exception("export failed")
                warnings.append(f"Dosya oluşturulamadı: {type(e).__name__}: {e}")
        else:
            warnings.append("Dosya istendi ama bu sohbette henüz dışa aktarılacak bir cevap yok.")
    full = AskResponse(answer=r["answer"], sources=sources, files=files,
                       tickets_opened=r["tickets_opened"],
                       tools_used=r["tools_used"], warnings=warnings, steps=r["steps"],
                       trace=r["trace"], standalone_question=r.get("standalone_question"),
                       timeline=[TimelineStep(**x) for x in r.get("timeline", [])],
                       usage=_usage(m, r["latency_ms"]))
    if key and not files and not r["export_request"]:
        with _ans_lock:
            ANSWERS[key] = full.model_copy(deep=True)
            while len(ANSWERS) > MAX_ANSWERS:
                ANSWERS.popitem(last=False)
    if not req.debug:
        full.trace = None
    return full


@app.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
def ask(req: AskRequest):
    return run_ask(req)


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """Aynı iş, ama ilerleme adımları anında gelir: her satır bir JSON (NDJSON).
    {"type":"step","text":...} ... sonra {"type":"result","data":AskResponse} ya da {"type":"error",...}.
    Agent ayrı bir thread'de çalışır, adımları kuyruğa yazar; bu uç kuyruğu okuyup akıtır."""
    q: "queue.Queue[dict | None]" = queue.Queue()

    def worker():
        try:
            res = run_ask(req, on_event=lambda text: q.put({"type": "step", "text": text}))
            q.put({"type": "result", "data": res.model_dump(mode="json", exclude_none=True)})
        except llm.LLMError as e:
            q.put({"type": "error", "status": e.status, "error": e.code, "detail": str(e)})
        except Exception as e:                          # akış başladıktan sonra HTTP kodu değişemez
            log.exception("stream failed")
            q.put({"type": "error", "status": 500, "error": "internal", "detail": f"{type(e).__name__}"})
        finally:
            q.put(None)
    threading.Thread(target=worker, daemon=True).start()

    def lines():
        while (item := q.get()) is not None:
            yield json.dumps(item, ensure_ascii=False) + "\n"
    return StreamingResponse(lines(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _source_files() -> dict[str, str]:
    """Manifest'teki kaynak adı → dosya adı (ör. wikipedia → Dinozorlar.pdf)."""
    import json
    s = get_settings()
    return {x["source"]: Path(x["path"]).name
            for x in json.loads(s.sources_path.read_text(encoding="utf-8"))}


def _sources_meta(answer, links: dict | None = None) -> dict:
    """Cevapta geçen her chunk id'si için dosya + sayfa. Tıklanınca PDF o sayfada açılır.
    web:/ref: kaynakları (canlı Wikipedia, referans nesne) gerçek URL'ye bağlanır."""
    ret, files, out = app.state.retriever, _source_files(), {}
    for c in answer.document + answer.inference + list(getattr(answer, "web_facts", [])):
        for sid in c.sources:
            if sid in (links or {}):
                kind = "Canlı Wikipedia" if sid.startswith("web:") else "Referans (genel bilgi)"
                out[sid] = SourceInfo(doc=sid.split(":", 1)[1], section=kind, url=links[sid])
                continue
            ch = ret.by_id.get(sid)
            if ch:
                pages = ch.get("pages") or []
                out[sid] = SourceInfo(doc=ch["doc"], section=ch.get("section", ""),
                                      file=files.get(ch.get("source")), page=pages[0] if pages else None)
    return out


@app.get("/pdf/{name}", include_in_schema=False)
def pdf(name: str):
    """Sadece manifest'te kayıtlı PDF'ler verilir (yol oyunlarıyla başka dosya okunamaz)."""
    from fastapi import HTTPException
    s = get_settings()
    allowed = {Path(p).name: p for p in _source_files().values()}
    if name not in allowed:
        raise HTTPException(404, "Böyle bir kaynak yok")
    return FileResponse(s.sources_path.parent / name, media_type="application/pdf")


@app.get("/exports/{name}", include_in_schema=False)
def download(name: str):
    """Sadece bizim ürettiğimiz isim kalıbı kabul edilir (dino_<tarih>_<id>.uzantı): başka dosya okunamaz."""
    from fastapi import HTTPException
    path = export.export_dir() / name
    if not export.NAME_RE.match(name) or not path.is_file():
        raise HTTPException(404, "Dosya bulunamadı")
    media = {"pdf": "application/pdf",
             "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
             "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}[path.suffix[1:]]
    return FileResponse(path, media_type=media, filename=name)


@app.post("/check", response_model=CheckResponse)
def check(req: CheckRequest):
    with llm.meter() as m:
        r = factcheck.check(req.claim, app.state.retriever)
    return CheckResponse(result=r["result"], reliable=r["reliable"], retrieved=r["retrieved"],
                         warnings=r["warnings"], usage=_usage(m, r["latency_ms"]))


@app.get("/tickets", response_model=list[Ticket])
def tickets(status: str | None = Query(None, description="open / closed")):
    rows = read_tickets(get_settings().tickets_path)
    return [t for t in rows if status is None or t["status"] == status]
