"""Agent'ın kullanabildiği araçlar.

Her araç = (argüman şeması, fonksiyon, tür). Model sadece argümanları üretir, fonksiyonu HER ZAMAN
bizim kodumuz çalıştırır. Üç tür araç var:

  read     search_docs, find_species   bilgi okur, hiçbir şeyi değiştirmez
  compute  compare_timeline            LLM'in güvenilmez olduğu işi (tarih aritmetiği) kod yapar
  write    open_gap_ticket             dosyaya kayıt yazar → ek kurallar: tekrar yok, istek başı 1

Tasarım kuralları (el kitabı Bölüm 4):
- Ham SQL yerine TİPLİ parametreler: model izin verilenin dışına çıkamaz, doğrulama kolay.
- Hata durumunda istisna fırlatmak yerine modele OKUNABİLİR bir hata mesajı dönülür;
  model düzeltip tekrar deneyebilir. Hata mesajı ne yapması gerektiğini söyler.
"""
import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal, Optional

import pandas as pd
from openai import pydantic_function_tool
from pydantic import BaseModel, Field, ValidationError, field_validator

from .config import get_settings


# ------------------------------------------------------------------------------------------
# Bağlam: bir isteğin araçlar arasında paylaştığı durum
# ------------------------------------------------------------------------------------------
@dataclass
class ToolContext:
    retriever: object = None                      # retrieval.Retriever
    tickets_path: Path | None = None
    max_tickets: int = 1
    opened_tickets: list[str] = field(default_factory=list)
    seen_chunks: list[str] = field(default_factory=list)   # agent'ın gördüğü chunk id'leri
    tool_outputs: list[str] = field(default_factory=list)  # yargıç için: araçlardan gelen her şey
    calls: set = field(default_factory=set)                # aynı çağrı tekrarını yakalamak için
    searched: bool = False                                 # bu istekte belgeler arandı mı (canlı arama kapısı)
    pending_export: dict | None = None                     # model dosya istedi; dosyayı cevaptan SONRA kod üretir
    allow_files: bool = False                              # dosya araçları sadece oturumlu sohbette
    files: list = field(default_factory=list)              # bu istekte üretilen dosyalar (indirme linki)
    on_event: Callable[[str], None] | None = None           # ilerleme bildirimi (arayüzdeki adım satırı)


# ------------------------------------------------------------------------------------------
# Tür tablosu
# ------------------------------------------------------------------------------------------
NUM_COLS = ["start_ma", "end_ma", "length_m_min", "length_m_max", "mass_t_min", "mass_t_max"]


@lru_cache
def species_table() -> pd.DataFrame:
    """İki kaynaktan birleşik tablo:
       guide      data/species.csv: rehberin Tablo 1'i, elle kontrol edildi (öncelikli)
       wikipedia  data/taxa_wiki.csv: `python -m dino.extract` ile LLM'in çıkardığı değerler
    Aynı tür iki kez görünebilir (farklı değerlerle). Bu bilerek: kaynak çatışması görünür kalsın."""
    s = get_settings()
    frames = [pd.read_csv(s.species_csv, keep_default_na=False).assign(origin="guide")]
    wiki = s.species_csv.parent / "taxa_wiki.csv"
    if wiki.exists():
        frames.append(pd.read_csv(wiki, keep_default_na=False).assign(origin="wikipedia"))
    df = pd.concat(frames, ignore_index=True)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")       # boş hücre → NaN
    for c in ["aliases", "period", "diet", "region", "section"]:
        df[c] = df[c].fillna("") if c in df else ""
    df["mass_mid"] = (df.mass_t_min + df.mass_t_max) / 2
    df["length_mid"] = (df.length_m_min + df.length_m_max) / 2
    return df


def _rng(a, b) -> Optional[str]:
    if pd.isna(a) or pd.isna(b):
        return None                                          # bilinmiyor: uydurma
    a, b = float(a), float(b)
    fmt = lambda x: f"{x:g}"
    return fmt(a) if a == b else f"{fmt(a)}-{fmt(b)}"


def _row(r) -> dict:
    return {"name": r["name"], "period": r["period"] or None,
            "age_ma": _rng(r.start_ma, r.end_ma),
            "length_m": _rng(r.length_m_min, r.length_m_max),
            "mass_t": _rng(r.mass_t_min, r.mass_t_max), "diet": r.diet or None,
            "region": r.region or None, "source": r.section, "origin": r.origin,
            **({"url": r["url"]} if isinstance(r.get("url"), str) and r.get("url") else {})}


# Gündelik referans nesneler (A5). Belge değil GENEL BİLGİ: origin='reference' ile işaretli, her biri
# kaynak olarak bir Wikipedia sayfasına bağlanır. Böylece "Argentinosaurus kaç fil?" sorusu kodla hesaplanır.
REFERENCE = [
    # name, aliases, start_ma, end_ma, len_min, len_max, mass_min, mass_max, wikipedia sayfası
    ("Human", "humans|insan|person|homo sapiens", 0.3, 0, 1.6, 1.8, 0.06, 0.08, "Human"),
    ("African elephant", "elephant|fil|african bush elephant", None, None, 6.0, 7.5, 4.0, 6.5,
     "African_bush_elephant"),
    ("City bus", "bus|otobüs|otobus", None, None, 12.0, 12.0, None, None, "Bus"),
    ("House cat", "cat|kedi|domestic cat", None, None, 0.46, 0.46, 0.004, 0.005, "Cat"),
    ("Chicken", "tavuk|hen", None, None, 0.4, 0.5, 0.002, 0.003, "Chicken"),
    ("Blue whale", "whale|mavi balina|balina", None, None, 24.0, 30.0, 100.0, 150.0, "Blue_whale"),
]


def _reference_rows() -> list[pd.Series]:
    out = []
    for name, al, s, e, l0, l1, m0, m1, page in REFERENCE:
        out.append(pd.Series({"name": name, "aliases": al, "period": "", "start_ma": s, "end_ma": e,
                              "length_m_min": l0, "length_m_max": l1, "mass_t_min": m0, "mass_t_max": m1,
                              "diet": "", "region": "", "section": "ref:" + name.lower().replace(" ", "_"),
                              "origin": "reference", "url": f"https://en.wikipedia.org/wiki/{page}"}))
    return out


def _norm_name(n: str) -> str:
    return re.sub(r"[^a-z]", "", n.lower().replace("ı", "i").replace("ü", "u").replace("ö", "o"))


def lookup_species(name: str, live: bool | None = None, ctx: "ToolContext | None" = None,
                   exact: bool = False) -> Optional[pd.Series]:
    """'T. rex', 'trex', 'Tyrannosaurus' → aynı satır. Tür iki kaynakta varsa rehberdeki döner.
    live_tools açıksa sıra: rehber → çıkarılmış tablo → referans nesneler → CANLI Wikipedia."""
    key = re.sub(r"[^a-z]", "", name.lower())
    if not key:
        return None
    found = _lookup_table(name, key, exact)
    if found is not None:
        return found
    live = get_settings().live_tools if live is None else live
    if not live:
        return None
    for r in _reference_rows():
        if any(_norm_name(n) == _norm_name(name) for n in [r["name"]] + r["aliases"].split("|")):
            return r
    if ctx is not None and not ctx.searched:
        # Önce belge, sonra web: canlı aramaya ancak bu istekte belgeler arandıktan sonra izin var.
        # (İlk denemede model belgelere hiç bakmadan Wikipedia'ya gidip "The documents show..." demişti.)
        _LIVE_ERRORS[key] = (f"'{name}' is not in the table. First search the documents for it with "
                             "search_docs. If they do not give the value, call this tool again: it will "
                             "then be looked up LIVE on Wikipedia.")
        return None
    from .wiki_live import lookup_taxon
    if ctx is not None and ctx.on_event:
        ctx.on_event(f"Wikipedia'dan canlı alınıyor: {name}")
    row = lookup_taxon(name)
    if "error" in row:
        _LIVE_ERRORS[key] = row["error"]
        return None
    return pd.Series({"aliases": "", "period": "", "diet": "", "region": "", **row})


_LIVE_ERRORS: dict[str, str] = {}


def _lookup_table(name: str, key: str, exact: bool = False) -> Optional[pd.Series]:
    for _, r in species_table().iterrows():
        names = [r["name"]] + [a for a in r.aliases.split("|") if a]
        if any(re.sub(r"[^a-z]", "", n.lower()) == key for n in names):
            return r
    if exact:                                               # tablo aracı: "Ceratops" ≠ "Triceratops"
        return None
    for _, r in species_table().iterrows():                 # kısmi eşleşme: "brachio"
        if key in re.sub(r"[^a-z]", "", r["name"].lower()):
            return r
    return None


def species_in_text(text: str) -> list[pd.Series]:
    """Metinde geçen türler (fact-check'te tablo satırlarını bağlama eklemek için)."""
    low = text.lower()
    out, seen = [], set()
    for _, r in species_table().iterrows():
        if r["name"] in seen:                               # aynı tür iki kaynakta: ilki (rehber)
            continue
        names = [r["name"]] + [a for a in r.aliases.split("|") if a]
        if any(re.search(rf"\b{re.escape(n.lower())}", low) for n in names):
            out.append(r)
            seen.add(r["name"])
    return out


KNOWN = lambda: ", ".join(sorted(set(species_table()["name"])))


def _missing_hint(missing: list[str]) -> str:
    errs = [_LIVE_ERRORS.get(re.sub(r"[^a-z]", "", m.lower())) for m in missing]
    errs = [e for e in errs if e]
    if errs:
        return " ".join(errs)
    return f"Use one of: {KNOWN()}."


# ------------------------------------------------------------------------------------------
# 1) search_docs  (read)
# ------------------------------------------------------------------------------------------
class SearchDocs(BaseModel):
    """Search the documents: the dinosaur field guide (source='guide') and 31 Wikipedia articles
    (source='wikipedia'). Use for biology, behavior, extinction, history, debates and myths.
    When a question asks for a number that the sources might disagree on, search each source
    separately and report both values."""
    query: str = Field(description="A search query in English, using scientific vocabulary")
    source: Optional[Literal["guide", "wikipedia"]] = Field(
        None, description="Restrict to one source; null = all")
    article: Optional[str] = Field(None, description="Restrict to one Wikipedia article; null = all")


def search_model(articles: list[str]) -> type[BaseModel]:
    """article alanını gerçek makale adlarıyla sınırlayan şema. Liste indeksten gelir:
    yeni makale eklenince şema kendiliğinden güncellenir, kodda elle liste tutulmaz."""
    from pydantic import create_model
    return create_model("SearchDocs", __base__=SearchDocs,
                        article=(Optional[Literal[tuple(articles)]],
                                 Field(None, description="Restrict to one Wikipedia article; null = all")))


def search_docs(a: SearchDocs, ctx: ToolContext) -> str:
    ctx.searched = True
    ids = ctx.retriever.search(a.query, source=a.source, doc=a.article, rewrite_mode="auto")
    extra = []
    if (a.source or a.article) and get_settings().search_spillover:
        # A1: filtre bilgiyi saklayamasın. İlk eval'de model aramayı tek makaleye daraltıp başka
        # dokümandaki cevabı kaçırıyordu (q15, q45, q52, q55). Filtre dışından en iyi 2 chunk eklenir.
        extra = [i for i in ctx.retriever.search(a.query, k=4, rewrite_mode="auto") if i not in ids][:2]
    if not ids and not extra:
        return json.dumps({"error": "No results with these filters. Try without 'article' or 'source'."})
    ctx.seen_chunks += [i for i in ids + extra if i not in ctx.seen_chunks]
    text = ctx.retriever.format(ids) if ids else "(no results inside the filter)"
    if extra:
        text += "\n\n--- Also relevant, from OUTSIDE your filter ---\n\n" + ctx.retriever.format(extra)
    return text


# ------------------------------------------------------------------------------------------
# 2) find_species  (read) — ham SQL yerine tipli filtre
# ------------------------------------------------------------------------------------------
class FindSpecies(BaseModel):
    """Filter and sort Table 1, the 10 profiled species. Use for sizes, masses, ages, diets and
    regions, and to rank or list species by these values. To COMPARE two taxa (bigger, how many
    times, same time?) use compare_size / compare_timeline instead: never compare rows yourself. Ranges like 6-12 are kept as ranges;
    sorting uses the midpoint. Rows come from the guide's table (origin='guide') and from values
    extracted from Wikipedia (origin='wikipedia'); the same taxon may appear twice with different
    values: report both. A null value means unknown, never guess it."""
    name: Optional[str] = Field(None, description="Part of a species name, e.g. 'rex'")
    diet: Optional[Literal["Carnivore", "Herbivore", "Omnivore", "Piscivore"]] = None
    period: Optional[Literal["Late Jurassic", "Mid Cretaceous", "Late Cretaceous"]] = None
    region: Optional[str] = Field(None, description="e.g. 'North America', 'Mongolia'")
    min_mass_t: Optional[float] = Field(None, description="Minimum mass in metric tonnes")
    max_mass_t: Optional[float] = Field(None, description="Maximum mass in metric tonnes")
    sort_by: Optional[Literal["mass_t", "length_m", "age_ma"]] = None
    descending: bool = Field(True, description="True = largest/oldest first")
    limit: int = Field(5, description="Number of rows, 1-10")

    @field_validator("limit")
    @classmethod
    def _limit(cls, v):
        if not 1 <= v <= 10:
            raise ValueError("limit must be between 1 and 10")
        return v


def find_species(a: FindSpecies, ctx: ToolContext) -> dict:
    df = species_table()
    if a.name:
        df = df[df["name"].str.contains(a.name, case=False, regex=False)]
    if a.diet:
        df = df[df.diet == a.diet]
    if a.period:
        df = df[df.period == a.period]
    if a.region:
        df = df[df.region.str.contains(a.region, case=False, regex=False)]
    if a.min_mass_t is not None:
        df = df[df.mass_t_max >= a.min_mass_t]
    if a.max_mass_t is not None:
        df = df[df.mass_t_min <= a.max_mass_t]
    if a.sort_by:
        col = {"mass_t": "mass_mid", "length_m": "length_mid", "age_ma": "start_ma"}[a.sort_by]
        df = df.dropna(subset=[col]).sort_values(col, ascending=not a.descending)
    rows = [_row(r) for _, r in df.head(a.limit).iterrows()]
    out = {"rows": rows, "total_matches": int(len(df)), "table": "Table 1 (section 4)"}
    if not rows and a.name and get_settings().live_tools:
        # Tabloda yoksa: compare araçlarıyla aynı zincir (referans → CANLI Wikipedia).
        # İlk koşuda model Ceratosaurus'u find_species ile 3 kez aradı, boş döndü, adım sınırına takıldı.
        r = lookup_species(a.name, ctx=ctx)
        if r is not None:
            return {"rows": [_row(r)], "total_matches": 1, "links": _links(r),
                    "table": "not in table: looked up live" if r.origin != "reference" else "reference"}
        out["note"] = f"No match for '{a.name}'. {_missing_hint([a.name])}"
    elif not rows:
        out["note"] = f"No species match these filters. The table only has: {KNOWN()}."
    return out


# ------------------------------------------------------------------------------------------
# 3) compare_timeline  (compute) — tarih aritmetiğini model değil kod yapar
# ------------------------------------------------------------------------------------------
class CompareTimeline(BaseModel):
    """Compare WHEN two profiled species lived: whether they lived at the same time and how many
    million years separate them. Always use this instead of comparing dates yourself."""
    species_a: str
    species_b: str


def compare_timeline(a: CompareTimeline, ctx: ToolContext) -> dict:
    ra, rb = lookup_species(a.species_a, ctx=ctx), lookup_species(a.species_b, ctx=ctx)
    missing = [n for n, r in [(a.species_a, ra), (a.species_b, rb)] if r is None]
    if missing:
        return {"error": f"Unknown species: {missing}. {_missing_hint(missing)}"}
    unknown = [r["name"] for r in (ra, rb) if pd.isna(r.start_ma) or pd.isna(r.end_ma)]
    if unknown:
        return {"error": f"Age unknown for {unknown} in the table. Search the documents instead."}
    # Ma: büyük sayı = daha eski. Aralık [end, start]
    gap = max(ra.end_ma - rb.start_ma, rb.end_ma - ra.start_ma, 0)
    overlap = max(min(ra.start_ma, rb.start_ma) - max(ra.end_ma, rb.end_ma), 0)
    same = gap == 0
    if same:
        text = (f"{ra['name']} ({_rng(ra.start_ma, ra.end_ma)} Ma) and {rb['name']} "
                f"({_rng(rb.start_ma, rb.end_ma)} Ma) lived at the same time.")
    else:
        older, younger = (ra, rb) if ra.start_ma > rb.start_ma else (rb, ra)
        text = (f"{older['name']} disappeared about {gap:g} million years before "
                f"{younger['name']} appeared; they never lived at the same time.")
    return {"links": _links(ra, rb), "species_a": _row(ra), "species_b": _row(rb), "lived_at_same_time": bool(same),
            "gap_million_years": float(gap), "overlap_million_years": float(overlap),
            "summary": text, "sources": [ra.section, rb.section], "computed_by": "code"}


# ------------------------------------------------------------------------------------------
# 4) compare_size  (compute) — oranı model değil kod hesaplar, aralıklar korunur
# ------------------------------------------------------------------------------------------
class CompareSize(BaseModel):
    """Compare the mass or length of two taxa from the table: how many times heavier/longer.
    Always use this instead of dividing numbers yourself. Ranges stay ranges (e.g. 8-10 times)."""
    taxon_a: str
    taxon_b: str
    measure: Literal["mass", "length"]


def _links(*rows) -> dict:
    """Canlı Wikipedia / referans satırları için kaynak id → gerçek URL (arayüz tıklanabilir yapar)."""
    return {r.section: r["url"] for r in rows if isinstance(r.get("url"), str) and r.get("url")}


def _ratio(a_min, a_max, b_min, b_max) -> tuple[float, float]:
    return a_min / b_max, a_max / b_min          # en küçük ve en büyük olası oran


def compare_size(a: CompareSize, ctx: ToolContext) -> dict:
    ra, rb = lookup_species(a.taxon_a, ctx=ctx), lookup_species(a.taxon_b, ctx=ctx)
    missing = [n for n, r in [(a.taxon_a, ra), (a.taxon_b, rb)] if r is None]
    if missing:
        return {"error": f"Unknown taxon: {missing}. {_missing_hint(missing)}"}
    lo_c, hi_c, unit = (("mass_t_min", "mass_t_max", "t") if a.measure == "mass"
                        else ("length_m_min", "length_m_max", "m"))
    vals = [ra[lo_c], ra[hi_c], rb[lo_c], rb[hi_c]]
    if any(pd.isna(v) for v in vals):
        return {"error": f"{a.measure} unknown for one of the taxa in the table. Search the documents."}
    if min(vals[2:]) <= 0:
        return {"error": "Invalid zero value in table."}
    lo, hi = _ratio(*vals)
    word = "heavier" if a.measure == "mass" else "longer"
    if lo < 0.9 and hi > 1.1:
        # Aralıklar örtüşüyor: hangisinin büyük olduğu veriyle söylenemez (Giganotosaurus 6-10 t vs T. rex 8 t)
        return {"bigger": None, "comparable": True,
                "values": {ra["name"]: f"{_rng(ra[lo_c], ra[hi_c])} {unit}",
                           rb["name"]: f"{_rng(rb[lo_c], rb[hi_c])} {unit}"},
                "summary": (f"The {a.measure} ranges overlap: {ra['name']} was {lo:.1f}-{hi:.1f} times "
                            f"{rb['name']}. They were comparable; the data cannot say which was {word}."),
                "sources": [ra.section, rb.section], "origins": [ra.origin, rb.origin],
                "links": _links(ra, rb), "computed_by": "code"}
    big, small, flip = (ra, rb, False) if (lo + hi) / 2 >= 1 else (rb, ra, True)
    if flip:
        lo, hi = 1 / hi, 1 / lo
    fmt = lambda x: f"{x:.0f}" if x >= 20 else f"{x:.1f}"     # 3250 kat / 8.1 kat
    times = fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)}-{fmt(hi)}"
    return {"bigger": big["name"], "smaller": small["name"], "times": times,
            "values": {ra["name"]: f"{_rng(ra[lo_c], ra[hi_c])} {unit}",
                       rb["name"]: f"{_rng(rb[lo_c], rb[hi_c])} {unit}"},
            "summary": f"{big['name']} was about {times} times {word} than {small['name']}.",
            "sources": [ra.section, rb.section], "origins": [ra.origin, rb.origin],
            "links": _links(ra, rb), "computed_by": "code"}


# ------------------------------------------------------------------------------------------
# 5) open_gap_ticket  (write) — bilgi tabanı eksik kaydı
# ------------------------------------------------------------------------------------------
class OpenGapTicket(BaseModel):
    """Record that the field guide does NOT contain the answer to a user's question, so the
    content team can add it. Use ONLY after searching the guide, ONLY when the answer is missing,
    and at most once per question."""
    question: str = Field(description="The user's question, as asked")
    missing_topic: str = Field(description="Short English name of the missing topic, "
                                           "e.g. 'Triceratops skin color'")
    priority: Literal["low", "medium", "high"] = Field(
        description="high = a basic question many users will ask; low = very niche")


_ticket_lock = threading.Lock()


def _ticket_key(topic: str) -> str:
    words = sorted(set(re.findall(r"[a-z0-9]+", topic.lower())) - {"the", "a", "of", "and"})
    return " ".join(words)


def read_tickets(path: Path) -> list[dict]:
    if not path or not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def open_gap_ticket(a: OpenGapTicket, ctx: ToolContext) -> dict:
    if len(ctx.opened_tickets) >= ctx.max_tickets:
        return {"error": "A ticket was already opened for this question. Do not open another."}
    path = ctx.tickets_path or get_settings().tickets_path
    key = _ticket_key(a.missing_topic)
    with _ticket_lock:                                  # aynı anda iki istek aynı numarayı almasın
        tickets = read_tickets(path)
        for t in tickets:
            if t["key"] == key and t["status"] == "open":
                ctx.opened_tickets.append(t["id"])
                return {"status": "already_open", "ticket_id": t["id"]}
        tid = f"T-{len(tickets) + 1:04d}"
        row = {"id": tid, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "question": a.question, "missing_topic": a.missing_topic, "priority": a.priority,
               "status": "open", "key": key}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    ctx.opened_tickets.append(tid)
    return {"status": "created", "ticket_id": tid}


# ------------------------------------------------------------------------------------------
# 6) export_document  (write / eylem) — model karar verir, dosyayı KOD üretir
# ------------------------------------------------------------------------------------------
class ExportDocument(BaseModel):
    """Create a downloadable file (PDF, Word or Excel) of the answers in this conversation. Use ONLY
    when the user explicitly asks for a file, document, report, export or download (e.g. 'bunu PDF
    yap', 'Excel'e aktar', 'save this as Word'). You do NOT write the file content: code builds it
    from the verified answers and their sources. If the same message also asks a question, answer it
    normally with the other tools too; the file will include that answer."""
    format: Literal["pdf", "docx", "xlsx"] = Field(description="pdf = PDF, docx = Word, xlsx = Excel")
    scope: Literal["last_answer", "conversation"] = Field(
        description="last_answer = only the latest answer; conversation = every answer so far")
    title: str = Field(description="Short document title in the user's language, e.g. "
                                   "'Ceratosaurus ve Allosaurus karşılaştırması'")


def export_document(a: ExportDocument, ctx: ToolContext) -> dict:
    if ctx.pending_export:
        return {"error": "A file was already requested in this message. Do not request another."}
    ctx.pending_export = a.model_dump()
    return {"status": "scheduled", "format": a.format, "scope": a.scope,
            "note": "The file will be generated by code after your final answer and shown as a download "
                    "link. In the summary just say briefly (in the user's language) that the file is "
                    "ready; do not repeat its content."}


# ------------------------------------------------------------------------------------------
# 7) species_table_file  (compute + write) — çok türlü tablo: model İSİM verir, değerleri KOD bulur
# ------------------------------------------------------------------------------------------
# Jeolojik zaman çizelgesi (ICS, milyon yıl önce). Dönem adını model değil bu tablo verir.
GEO_PERIODS = [(251.9, 247.2, "Erken Triyas"), (247.2, 237.0, "Orta Triyas"), (237.0, 201.4, "Geç Triyas"),
               (201.4, 174.7, "Erken Jura"), (174.7, 161.5, "Orta Jura"), (161.5, 145.0, "Geç Jura"),
               (145.0, 100.5, "Erken Kretase"), (100.5, 66.0, "Geç Kretase")]


def period_of(start_ma, end_ma) -> str | None:
    """[end, start] aralığının değdiği dönemler: 68-66 → 'Geç Kretase'; 150-140 → 'Geç Jura – Erken Kretase'."""
    if pd.isna(start_ma) or pd.isna(end_ma):
        return None
    hit = [name for old, young, name in GEO_PERIODS if start_ma > young and end_ma < old]
    return hit[0] if len(hit) == 1 else (f"{hit[0]} – {hit[-1]}" if hit else None)


ORIGIN_TR = {"guide": "Rehber tablosu", "wikipedia": "Wikipedia belgesi (korpus)",
             "wikipedia-live": "Canlı Wikipedia", "reference": "Genel referans"}


class SpeciesTableFile(BaseModel):
    """Build a downloadable TABLE FILE of many dinosaur genera (name, geological period, age in
    million years, length, mass, source of every row). Use when the user asks for a table, list or
    Excel/Word/PDF of several dinosaurs, e.g. 'all ceratopsids with their periods as Excel'.
    FIRST find which genera the documents mention with search_docs, then pass their names here.
    You never write any values: code looks each genus up (guide table, values extracted from the
    documents, then live Wikipedia), computes the period from the ages and writes the file.
    Do not list the table in your answer; just summarise it."""
    names: list[str] = Field(description="Genus names, at most 30, e.g. ['Triceratops', 'Centrosaurus']")
    format: Literal["xlsx", "docx", "pdf"] = Field(description="xlsx = Excel (default), docx = Word, pdf = PDF")
    title: str = Field(description="Table title in the user's language, e.g. 'Ceratopsidae cinsleri'")


def species_table_file(a: SpeciesTableFile, ctx: ToolContext) -> dict:
    if not ctx.allow_files:
        return {"error": "Files can only be created in a chat session."}
    if not ctx.searched:
        # Önce belge: liste modelin ezberinden değil belgelerden gelsin.
        return {"error": "First use search_docs to find which genera the documents mention, then call "
                         "this tool again with those names."}
    names = list(dict.fromkeys(n.strip() for n in a.names if n.strip()))
    if not names or len(names) > 30:
        return {"error": "Pass between 1 and 30 genus names."}
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    def one(n):
        r = lookup_species(n, ctx=ctx, exact=True)
        return n, r, _LIVE_ERRORS.get(re.sub(r"[^a-z]", "", n.lower()))
    with ThreadPoolExecutor(max_workers=6) as ex:     # canlı aramalar paralel; maliyet sayacı korunur
        results = list(ex.map(lambda n: contextvars.copy_context().run(one, n), names))

    rows, missing, seen = [], [], set()
    for n, r, err in results:
        if r is None:
            missing.append({"Aranan": n, "Neden": err or "Tabloda ve Wikipedia'da bulunamadı."})
            continue
        if r["name"] in seen:                          # iki isim aynı cinse çıktı (eş anlamlı / yönlendirme)
            continue
        seen.add(r["name"])
        rows.append({"Cins": r["name"], "Dönem": period_of(r.start_ma, r.end_ma) or (r.get("period") or None),
                     "Milyon yıl önce": _rng(r.start_ma, r.end_ma),
                     "Uzunluk (m)": _rng(r.length_m_min, r.length_m_max),
                     "Ağırlık (t)": _rng(r.mass_t_min, r.mass_t_max),
                     "Kaynak türü": ORIGIN_TR.get(r.origin, r.origin),
                     "Kaynak": r["url"] if isinstance(r.get("url"), str) and r.get("url") else r.section})
    rows.sort(key=lambda x: (x["Dönem"] is None, x["Cins"]))
    if not rows:
        return {"error": "None of the names were found.", "missing": missing}
    from . import export
    path = export.render_table(a.title, rows, missing, a.format)
    ctx.files.append({"name": path.name, "format": a.format, "turns": len(rows)})
    by_src = {k: sum(r["Kaynak türü"] == k for r in rows) for k in {r["Kaynak türü"] for r in rows}}
    return {"status": "created", "file": path.name, "rows": len(rows), "by_source": by_src,
            "not_found": [m["Aranan"] for m in missing],
            "links": {f"web:{r['Cins']}": r["Kaynak"] for r in rows if str(r["Kaynak"]).startswith("http")},
            "note": "The file is attached as a download link. In the summary say (in the user's language) "
                    "how many genera it has, where the values came from and which names were not found. "
                    "Do not list the table."}


# ------------------------------------------------------------------------------------------
# Kayıt defteri ve çalıştırıcı
# ------------------------------------------------------------------------------------------
@dataclass
class Tool:
    schema: type[BaseModel]
    fn: Callable
    kind: Literal["read", "compute", "write"]


TOOLS: dict[str, Tool] = {
    "search_docs": Tool(SearchDocs, search_docs, "read"),
    "find_species": Tool(FindSpecies, find_species, "read"),
    "compare_timeline": Tool(CompareTimeline, compare_timeline, "compute"),
    "compare_size": Tool(CompareSize, compare_size, "compute"),
    "open_gap_ticket": Tool(OpenGapTicket, open_gap_ticket, "write"),
    "export_document": Tool(ExportDocument, export_document, "write"),
    "species_table_file": Tool(SpeciesTableFile, species_table_file, "write"),
}


def schema_for(name: str, ctx: "ToolContext | None" = None) -> type[BaseModel]:
    if name == "search_docs" and ctx is not None and ctx.retriever is not None:
        return search_model(ctx.retriever.articles)        # makale listesi indeksten
    return TOOLS[name].schema


LIVE_NOTE = (" Any dinosaur genus works: if it is not in the table and search_docs did not give "
             "the value, it is looked up LIVE on Wikipedia (origin='wikipedia-live', with its URL). Everyday references also work: "
             "Human, African elephant, City bus, House cat, Chicken, Blue whale (origin='reference').")


def tool_specs(names=None, ctx: "ToolContext | None" = None) -> list[dict]:
    """OpenAI 'tools' listesi. Şema pydantic modelinden üretilir (strict mode): elle JSON yazmak yok.
    live_tools açıksa karşılaştırma araçlarının açıklamasına canlı arama yeteneği eklenir:
    model, aracın ne yapabildiğini SADECE açıklamadan öğrenir."""
    live = get_settings().live_tools
    def desc(n):
        d = TOOLS[n].schema.__doc__.strip()
        return d + LIVE_NOTE if live and n in ("compare_size", "compare_timeline", "find_species") else d
    return [pydantic_function_tool(schema_for(n, ctx), name=n, description=desc(n))
            for n in TOOLS if names is None or n in names]


def execute(name: str, arguments: str, ctx: ToolContext) -> str:
    """Model ne gönderirse göndersin buradan çökmeden bir metin döner."""
    tool = TOOLS.get(name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool '{name}'. Available: {list(TOOLS)}"})
    try:
        args = schema_for(name, ctx).model_validate_json(arguments or "{}")
    except ValidationError as e:                          # strict mode'a rağmen ikinci kontrol
        errs = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
        return json.dumps({"error": f"Invalid arguments: {errs}. Fix them and call again."})
    # Aynı çağrıyı tekrar etmek adım yakar. Ama belgeler arandıktan sonra durum değişir
    # (canlı arama açılır), o yüzden 'searched' da anahtarın parçası: arama öncesi kapıya takılan
    # çağrı, aramadan sonra tekrar yapılabilir.
    key = (name, args.model_dump_json(), ctx.searched)
    if key in ctx.calls:
        return json.dumps({"error": "You already made this exact call; its result is above. "
                           "Do not repeat it: use a different tool or answer with what you have."})
    ctx.calls.add(key)
    try:
        out = tool.fn(args, ctx)
    except Exception as e:                                # araç hatası agent'ı öldürmesin
        return json.dumps({"error": f"Tool failed: {type(e).__name__}: {e}"})
    text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
    ctx.tool_outputs.append(f"[{name}] {text}")
    return text
