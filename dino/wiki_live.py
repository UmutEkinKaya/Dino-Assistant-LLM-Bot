"""Canlı Wikipedia: korpusta olmayan bir cinsin boyut/yaş bilgisini anında getirir.

Kullanım (kod içinden; agent bunu compare_size / compare_timeline üzerinden tetikler):
    lookup_taxon("Ceratosaurus")  →  {"name", "length_m_min", ..., "url", "origin": "wikipedia-live"}

Tasarım kararları (el kitabı Bölüm 12):
  • Sadece Wikipedia API'si (allowlist). Genel web araması yok: güvenilmez kaynak + sayfaya
    gömülü "talimat" metinleri (prompt injection) riski. Gelen metin VERİDİR, talimat değil.
  • Model URL yazmaz. Sayfa adını kod sorgular; URL'yi API döndürür (gerçek, doğrulanmış).
  • Getirilen metin, offline çıkarımla AYNI hattan geçer (extract.py): structured output +
    "alıntı metinde var mı / sayı alıntıda var mı / birim dönüşümü kodda" kontrolleri.
  • Önbellek: aynı cins ikinci kez sorulursa ağa ve LLM'e gidilmez (data/live_cache.jsonl).
  • Ağ yoksa / sayfa yoksa çökmez; modele okunabilir bir hata döner.

Ofis ağında SSL denetimi varsa: `pip install truststore` → Windows sertifika deposu kullanılır.
"""
import json
import re
import threading
from datetime import datetime, timezone

from .config import get_settings

ALLOWED_HOSTS = {"en.wikipedia.org"}
_lock = threading.Lock()


def _cache_path():
    return get_settings().species_csv.parent / "live_cache.jsonl"


def _http_get(params: dict) -> dict:
    import requests
    try:                                                   # şirket SSL denetimi: OS sertifika deposu
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    base = get_settings().wiki_api_base.rstrip("/")
    host = re.sub(r"^https?://", "", base).split("/")[0].split(":")[0]
    if host not in ALLOWED_HOSTS and host not in ("127.0.0.1", "localhost"):
        raise ValueError(f"İzin verilmeyen adres: {host}")
    r = requests.get(base + "/w/api.php", params=params, timeout=10,
                     headers={"User-Agent": "DinoBilgiAsistani/1.0 (egitim projesi)"})
    r.raise_for_status()
    return r.json()


def fetch_page(title: str) -> dict | None:
    """Sayfanın düz metni + gerçek URL. Yönlendirmeleri izler (ör. 'T. rex' → Tyrannosaurus)."""
    data = _http_get({"action": "query", "prop": "extracts|info", "explaintext": 1, "redirects": 1,
                      "inprop": "url", "format": "json", "titles": title})
    for page in data.get("query", {}).get("pages", {}).values():
        if "missing" in page or not page.get("extract"):
            return None
        return {"title": page["title"], "url": page.get("fullurl", ""), "text": page["extract"]}
    return None


def _chunks(text: str, size: int = 1200) -> list[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip() and not p.startswith("==")]
    out, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > size:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n{p}".strip()
    if cur:
        out.append(cur)
    return out


def _read_cache() -> dict:
    p = _cache_path()
    if not p.exists():
        return {}
    from .extract import PROMPT_VERSION
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["key"]: r for r in rows if r.get("prompt") == PROMPT_VERSION}


def lookup_taxon(name: str, max_chunks: int = 5) -> dict:
    """Cinsi Wikipedia'dan getir, boyut/yaşı çıkar, doğrula. Her zaman dict döner (hata dahil)."""
    from . import llm
    from .extract import PROMPT_VERSION, SIZE_RE, SYSTEM, Extraction, aggregate, validate

    key = re.sub(r"[^a-z]", "", name.lower())
    cache = _read_cache()
    if key in cache:
        return cache[key]["row"]
    try:
        page = fetch_page(name)
    except Exception as e:                                 # ağ, SSL, timeout: çökme, söyle
        return {"error": f"Wikipedia'ya ulaşılamadı ({type(e).__name__}). "
                         "Answer from the documents only and say the live lookup failed."}
    if page is None:
        return {"error": f"No Wikipedia article named '{name}'. Check the spelling."}

    genus = page["title"].split()[0].strip(" ,.")
    parts = _chunks(page["text"])
    cand = [parts[0]] + [c for c in parts[1:] if SIZE_RE.search(c)]  # giriş + boyut geçen paragraflar
    rows = []
    for i, text in enumerate(cand[:max_chunks]):
        try:
            ex = llm.parse([{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": text}], Extraction, tag="live_extract").parsed
        except llm.LLMError:
            continue
        for f in ex.facts:
            row, _ = validate(f, text)
            if row and row["name"].lower() == genus.lower():  # sadece sayfanın kendi cinsi
                rows.append({**row, "chunk": f"{genus.lower()}:{i}"})
    if not rows:
        return {"error": f"The Wikipedia article '{page['title']}' gives no verifiable size or age "
                         "values.", "url": page["url"]}
    agg = aggregate(rows).iloc[0].to_dict()
    agg = {k: (None if isinstance(v, float) and v != v else v) for k, v in agg.items()}
    agg.update(section=f"web:{genus}", origin="wikipedia-live", url=page["url"],
               fetched=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with _lock, open(_cache_path(), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": key, "prompt": PROMPT_VERSION, "row": agg}, ensure_ascii=False) + "\n")
    return agg
