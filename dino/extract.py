"""Offline çıkarım: Wikipedia metninden boyut/yaş tablosu (structured output, ölçekte).

  python -m dino.extract            # ~110 LLM çağrısı, birkaç cent; sonuç data/taxa_wiki.csv
  python -m dino.extract --limit 5  # deneme

Neden agent aracı DEĞİL de ayrı bir adım? Sonuç her soruda aynı olur. Bir kez hesaplayıp
kaydetmek, her soruda LLM'e yaptırmaktan ucuz, hızlı ve tutarlı. Agent bu tabloyu
find_species / compare_size / compare_timeline üzerinden okur.

LLM'e güvenmiyoruz, kontrol ediyoruz (her biri kodda):
  1. Alıntı chunk'ta gerçekten geçiyor mu?             → geçmiyorsa satır atılır
  2. Her sayı alıntının içinde geçiyor mu?             → geçmiyorsa o alan atılır
  3. Birim dönüşümü (ft→m, kg→t) modelde değil kodda
  4. Makul aralık: uzunluk 0.1-60 m, kütle 0.0001-150 t, yaş 0-260 Ma
Sonunda rehberin elle kontrol edilmiş tablosuyla karşılaştırılır (uyum raporu).
"""
import argparse
import json
import re
from collections import defaultdict
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from . import llm
from .config import ROOT, get_settings
from .factcheck import verify_quote
from .retrieval import Retriever

SIZE_RE = re.compile(r"\d[\d.,–-]*\s?(?:m|metres|meters|kg|kilograms|tonnes|metric tons|t|ft|feet)\b(?!\w)")
SKIP_GROUPS = ("2. Mesozoik", "6. Toplu")               # jeoloji / yok oluş: tür boyutu yok
CACHE = ROOT / "data" / "extract_cache.jsonl"


class SizeFact(BaseModel):
    taxon: str = Field(description="Genus name only, e.g. 'Argentinosaurus' (no species epithet)")
    quote: str = Field(description="Exact words copied from the text that state these values, "
                                   "max 40 words")
    length_min: Optional[float] = None
    length_max: Optional[float] = None
    length_unit: Optional[Literal["m", "ft"]] = None
    mass_min: Optional[float] = None
    mass_max: Optional[float] = None
    mass_unit: Optional[Literal["kg", "t"]] = Field(None, description="t = metric tonnes")
    age_start_ma: Optional[float] = Field(None, description="Older bound, million years ago")
    age_end_ma: Optional[float] = Field(None, description="Younger bound, million years ago")


class Extraction(BaseModel):
    facts: list[SizeFact] = Field(description="One entry per genus whose ADULT length, mass or "
                                              "age is explicitly stated. Empty if none.")


# v2 (ilk eval'den): "Velociraptor 90 kg" filmdeki raptordan geldi; "Spinosaurus 20.9 t" uç bir
# tahminin en geniş aralığa girmesinden. Prompt ve birleştirme kuralı buna göre değişti.
PROMPT_VERSION = "v2"
SYSTEM = ("Extract body length, body mass and geological age for dinosaur genera from the text. "
          "Only values explicitly stated for that genus; never estimate or convert. Prefer metric "
          "values when both are given. If a single value is given, use it as both min and max. "
          "Skip juveniles, eggs, footprints, skulls and body parts. Skip values that describe "
          "films, novels, toys or other popular-culture depictions, and values given for a "
          "DIFFERENT genus used as a comparison (e.g. 'larger than the 90 kg animals in the films').")


def _nums(text: str) -> set[float]:
    out = set()
    for m in re.findall(r"\d+(?:[.,]\d+)*", text):
        try:
            out.add(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def validate(f: SizeFact, chunk_text: str) -> tuple[Optional[dict], list[str]]:
    """Bir çıkarımı kontrol edip temiz satıra çevirir. (satır | None, sorunlar)"""
    issues = []
    if not verify_quote(f.quote, chunk_text):
        return None, ["alıntı chunk'ta yok"]
    qn = _nums(f.quote)
    row = {"name": f.taxon.split()[0].strip(" ,.").capitalize()}

    def take(lo, hi, conv, lim, label):
        if lo is None and hi is None:
            return None, None
        lo, hi = (lo if lo is not None else hi), (hi if hi is not None else lo)
        if lo not in qn or hi not in qn:
            issues.append(f"{label}: sayı alıntıda yok")
            return None, None
        lo, hi = sorted((conv(lo), conv(hi)))
        if not (lim[0] <= lo and hi <= lim[1]):
            issues.append(f"{label}: makul aralık dışı ({lo}-{hi})")
            return None, None
        return round(lo, 4), round(hi, 4)

    to_m = (lambda x: x * 0.3048) if f.length_unit == "ft" else (lambda x: x)
    to_t = (lambda x: x / 1000) if f.mass_unit == "kg" else (lambda x: x)
    row["length_m_min"], row["length_m_max"] = take(f.length_min, f.length_max, to_m, (0.1, 60), "uzunluk")
    row["mass_t_min"], row["mass_t_max"] = take(f.mass_min, f.mass_max, to_t, (0.0001, 150), "kütle")
    a, b = take(f.age_start_ma, f.age_end_ma, lambda x: x, (0, 260), "yaş")
    row["start_ma"], row["end_ma"] = (b, a) if a is not None else (None, None)   # start = eski
    if all(row[k] is None for k in ["length_m_min", "mass_t_min", "start_ma"]):
        return None, issues or ["değer yok"]
    return row, issues


def candidates(ret: Retriever) -> list[dict]:
    out = []
    for c in ret.chunks:
        if c.get("source") != "wikipedia" or c["group"].startswith(SKIP_GROUPS):
            continue
        if c["kind"] == "infobox" or SIZE_RE.search(c["text"]):
            out.append(c)
    return out


def load_cache() -> dict:
    """Önbellek anahtarı chunk + PROMPT SÜRÜMÜ. Prompt değişince eski cevaplar kullanılmaz;
    yoksa yeni promptu test ettiğini sanıp eski sonuçlara bakarsın (sessiz hata)."""
    if not CACHE.exists():
        return {}
    rows = map(json.loads, CACHE.read_text(encoding="utf-8").splitlines())
    return {r["chunk"]: r for r in rows if r.get("prompt") == PROMPT_VERSION}


def aggregate(rows: list[dict]) -> pd.DataFrame:
    """Aynı cins için birden çok bulgu → tek satır.

    v1 her alanın EN GENİŞ aralığını alıyordu: tek bir uç değer (20.9 t) bütün satırı bozuyordu.
    v2: (1) cinsin KENDİ makalesinden gelen bulgular varsa sadece onlar kullanılır;
        (2) yoksa alt sınırların medyanı ve üst sınırların medyanı alınır (uç değere dayanıklı).
    """
    from statistics import median
    by = defaultdict(list)
    for r in rows:
        by[r["name"]].append(r)
    out = []
    for name, facts in by.items():
        own = [f for f in facts if f["chunk"].split(":")[0] == name.lower()]
        use = own or facts
        rec = {"name": name, "aliases": "", "period": "", "diet": "", "region": ""}
        for lo, hi in [("length_m_min", "length_m_max"), ("mass_t_min", "mass_t_max"),
                       ("end_ma", "start_ma")]:
            los = [f[lo] for f in use if f.get(lo) is not None]
            his = [f[hi] for f in use if f.get(hi) is not None]
            rec[lo] = round(median(los), 4) if los else None
            rec[hi] = round(median(his), 4) if his else None
            if rec[lo] is not None and rec[hi] is not None and rec[lo] > rec[hi]:
                rec[lo], rec[hi] = rec[hi], rec[lo]
        rec["section"] = "|".join(dict.fromkeys(f["chunk"] for f in use))
        out.append(rec)
    cols = ["name", "aliases", "period", "start_ma", "end_ma", "length_m_min", "length_m_max",
            "mass_t_min", "mass_t_max", "diet", "region", "section"]
    return pd.DataFrame(out, columns=cols).sort_values("name")


def agreement_report(wiki: pd.DataFrame) -> pd.DataFrame:
    """Rehberin elle kontrol edilmiş tablosuyla karşılaştırma. Aralıklar kesişiyorsa 'uyumlu'.
    Uyumsuzluk her zaman hata değildir: kaynaklar gerçekten farklı rakam verebilir. Satırı oku."""
    from .tools import lookup_species
    guide = pd.read_csv(get_settings().species_csv)
    rows = []
    for _, w in wiki.iterrows():
        g = lookup_species(w["name"])
        if g is None or g["origin"] != "guide":
            continue
        for label, lo, hi in [("uzunluk", "length_m_min", "length_m_max"),
                              ("kütle", "mass_t_min", "mass_t_max"), ("yaş", "end_ma", "start_ma")]:
            if pd.isna(w[lo]) or pd.isna(g[lo]):
                continue
            overlap = w[lo] <= g[hi] and g[lo] <= w[hi]
            # v1'de sadece "kesişiyor mu?" bakılıyordu: 14-90 kg da 15-20 kg ile "uyumlu" çıktı.
            # Artık Wikipedia aralığı rehberin yarısı ile iki katı arasında kalmalı.
            within = w[lo] >= g[lo] * 0.5 and w[hi] <= g[hi] * 2
            rows.append({"tür": g["name"], "alan": label, "rehber": f"{g[lo]:g}-{g[hi]:g}",
                         "wikipedia": f"{w[lo]:g}-{w[hi]:g}", "kesişiyor": overlap,
                         "makul_aralık": within, "uyumlu": overlap and within})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    s = get_settings()
    ret = Retriever.load()
    todo = candidates(ret)[:a.limit]
    cache = load_cache()
    stats = defaultdict(int)
    print(f"{len(todo)} aday chunk ({sum(c['id'] in cache for c in todo)} tanesi önbellekte)")
    with llm.meter() as m:
        for c in todo:
            if c["id"] not in cache:
                try:
                    ex = llm.parse([{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": c["text"]}], Extraction,
                                   tag="extract").parsed
                    facts = [f.model_dump() for f in ex.facts]
                except llm.LLMError as e:
                    print(f"  ! {c['id']}: {e.code}")
                    continue
                cache[c["id"]] = {"chunk": c["id"], "prompt": PROMPT_VERSION, "facts": facts}
                with open(CACHE, "a", encoding="utf-8") as fh:        # önbellek: tekrar ödeme yok
                    fh.write(json.dumps(cache[c["id"]], ensure_ascii=False) + "\n")
            print(".", end="", flush=True)
    rows = []
    for c in todo:
        for f in cache.get(c["id"], {}).get("facts", []):
            stats["çıkarılan"] += 1
            row, issues = validate(SizeFact(**f), c["text"])
            for i in issues:
                stats["sorun: " + i.split(" (")[0]] += 1
            if row:
                rows.append({**row, "chunk": c["id"]})
                stats["kabul"] += 1
    df = aggregate(rows)
    df.to_csv(s.species_csv.parent / "taxa_wiki.csv", index=False)
    print(f"\n\n{len(df)} cins → data/taxa_wiki.csv   (bu koşu: {m['calls']} çağrı, ${m['usd']:.4f})")
    for k, v in sorted(stats.items()):
        print(f"  {k:32s} {v}")
    from .tools import species_table
    species_table.cache_clear()
    rep = agreement_report(df)
    if len(rep):
        print(f"\nRehberle uyum: {rep['uyumlu'].mean():.0%} ({rep['uyumlu'].sum()}/{len(rep)})")
        print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
