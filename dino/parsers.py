"""Doküman ayrıştırıcıları: PDF → bölümler (section) → chunk'lar.

İki tür kaynak var:
  guide        Kendi rehberimiz (numaralı başlıklar, temiz metin)          → ingest.chunk_by_section
  wiki_bundle  Yer imleriyle birleştirilmiş Wikipedia PDF'leri (dağınık)  → bu dosya

Wikipedia PDF'lerinde gördüğümüz sorunlar ve çözümleri:
  1. Metnin ~%58'i kaynakça/notlar      → "References", "Notes"... başlıklarından sonrası atılır
  2. Makaleler farklı ölçekte basılmış  → punto MUTLAK değil, makalenin gövde puntosuna ORANLA okunur
     (çoğu makalede gövde 12pt, bazılarında 8pt; "başlık = 18pt" kuralı bunları sessizce kaçırır)
  3. Bilgi kutusu gövdeye karışıyor     → ilk ana başlıktan önceki sans-serif bloklar = bilgi kutusu
  4. Resim altı, soy ağacı şeması       → gövdeden küçük punto → atılır
  5. [12], [note 3], [a] işaretleri     → regex ile temizlenir

Önizleme:  python -m dino.parsers data/Dinozorlar.pdf
"""
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

STOP_SECTIONS = {"references", "notes", "sources", "bibliography", "further reading",
                 "external links", "citations", "footnotes", "see also", "cited sources",
                 "literature cited", "literature", "general and cited references", "works cited", "explanatory notes"}

CITE_RE = re.compile(r"\[\s*(?:\d+(?:[–,-]\d+)*|[a-z]{1,2}|note\s*\d+|nb\s*\d+|citation needed|"
                     r"clarification needed|page needed|which\?|when\?|who\?|dubious[^\]]*|"
                     r"[A-Z]\s*\d+)\s*\]")


@dataclass
class Section:
    doc: str                    # makale adı, ör. "Tyrannosaurus"
    group: str                  # derlemedeki bölüm, ör. "3. Saurischia (Kertenkele Kalçalılar)"
    path: list[str]             # ["Paleobiology", "Speed"]
    text: str
    pages: list[int] = field(default_factory=list)   # 1'den başlayan PDF sayfa numaraları
    kind: str = "body"          # body | infobox


def clean(t: str) -> str:
    t = t.replace("⁠", "").replace("­", "").replace("​", "")
    t = CITE_RE.sub("", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+([,.;:!?)\]])", r"\1", t)        # "( tyrannos , meaning" → "(tyrannos, meaning"
    t = re.sub(r"([(\[])\s+", r"\1", t)
    return t.strip()


def _block_text(b) -> str:
    lines = []
    for l in b["lines"]:
        lines.append("".join(sp["text"] for sp in l["spans"]))
    out = ""
    for ln in lines:                                    # satır sonu tirelemesini birleştir
        if out.endswith("-") and ln[:1].islower():
            out = out[:-1] + ln
        else:
            out = (out + " " + ln) if out else ln
    return out


def _dominant(b):
    """Bloğun karakter olarak en baskın span'i (font + punto)."""
    c = Counter()
    for l in b["lines"]:
        for sp in l["spans"]:
            if sp["text"].strip():
                c[(round(sp["size"], 1), sp["font"].split("+")[-1])] += len(sp["text"])
    return c.most_common(1)[0][0] if c else (0, "")


def _body_size(doc, start, end) -> float:
    c = Counter()
    for pno in range(start, end):
        for b in doc[pno].get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for sp in l["spans"]:
                    if "Serif" in sp["font"] and "Bold" not in sp["font"]:
                        c[round(sp["size"], 1)] += len(sp["text"])
    return c.most_common(1)[0][0] if c else 12.0


def parse_wiki_bundle(path: str) -> list[Section]:
    import pymupdf                                       # ağır bağımlılık: sadece burada
    doc = pymupdf.open(path)
    toc = doc.get_toc()                                  # [(seviye, başlık, sayfa(1'den))]
    boundaries = sorted({p - 1 for _, _, p in toc} | {len(doc)})
    sections: list[Section] = []
    group = ""
    for level, title, page in toc:
        if level == 1:
            group = title
            continue
        start = page - 1
        end = min(b for b in boundaries if b > start)
        body = _body_size(doc, start, end)
        sections += _parse_article(doc, title, group, start, end, body)
    return sections


def _parse_article(doc, title, group, start, end, body) -> list[Section]:
    out: list[Section] = []
    h2, h3 = "Overview", None
    seen_h2 = skipping = False
    buf, pages, infobox = [], [], []

    def flush():
        nonlocal buf, pages
        text = clean(" \n".join(buf))
        if len(text) > 40:
            out.append(Section(title, group, [h2] + ([h3] if h3 else []), text, sorted(set(pages))))
        buf, pages = [], []

    for pno in range(start, end):
        blocks = [b for b in doc[pno].get_text("dict")["blocks"] if "lines" in b]
        blocks.sort(key=lambda b: (round(b["bbox"][1]), b["bbox"][0]))   # yukarıdan aşağı
        for b in blocks:
            size, font = _dominant(b)
            if not size:
                continue
            r = size / body
            bold, serif = "Bold" in font, "Serif" in font
            raw = _block_text(b).strip()
            if not raw:
                continue
            # ---- başlıklar (oranla) ----
            if bold and serif and r >= 1.9:              # makale başlığı
                continue
            if bold and serif and 1.3 <= r < 1.9:        # ana başlık (H2)
                flush()
                h2, h3, seen_h2 = clean(raw), None, True
                skipping = h2.lower() in STOP_SECTIONS
                continue
            if bold and not serif and 1.1 <= r < 1.3 and len(raw) < 90:   # alt başlık (H3)
                if not skipping:
                    flush()
                h3 = clean(raw)
                continue
            if skipping:
                continue
            W = doc[pno].rect.width
            x0, x1 = b["bbox"][0], b["bbox"][2]
            side = (not serif) and x0 > W * 0.5 and (x1 - x0) < W * 0.45   # sağ kenar sütunu
            taxo = "Kingdom:" in raw or "Scientific classification" in raw
            # ---- bilgi kutusu: ilk ana başlıktan önceki sans-serif bloklar + sınıflandırma ----
            if (not seen_h2 and not serif) or taxo:
                if len(" ".join(infobox)) < 1500:
                    infobox.append(raw)
                continue
            if side:                                     # başlıktan sonra sağ sütun: resim altı vb.
                continue
            # ---- gürültü: resim altı yazısı, soy ağacı şeması (gövdeden küçük) ----
            if r < 0.93:
                continue
            # ---- gürültü: taksonomi listesi (satırların çoğu "†" ile başlıyor) ----
            lines = [l for l in ("".join(sp["text"] for sp in ln["spans"]).strip()
                                 for ln in b["lines"]) if l]
            if not serif and len(lines) >= 3 and sum(l.startswith("†") for l in lines) / len(lines) >= 0.5:
                continue
            buf.append(raw)
            pages.append(pno + 1)
    flush()
    if infobox:
        box = re.sub(r"(?:\b[A-Z][a-z]{0,2}\.?\s+){5,}", " ", " | ".join(infobox))  # "B V H B Apt." cetveli
        out.insert(0, Section(title, group, ["Infobox"], clean(box)[:1500],
                              [start + 1], kind="infobox"))
    return out


# ------------------------------------------------------------------------------------------
# Bölüm → chunk: paragraf sınırında böl, uzun paragrafı cümlelerden böl
# ------------------------------------------------------------------------------------------
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_text(text: str, max_chars: int = 1200, min_chars: int = 250) -> list[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    units = []
    for p in paras:
        if len(p) <= max_chars:
            units.append(p)
        else:
            units += SENT_RE.split(p)
    chunks, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) + 1 > max_chars:
            chunks.append(cur)
            cur = u
        else:
            cur = f"{cur} {u}".strip()
    if cur:
        if chunks and len(cur) < min_chars:             # çok kısa kuyruk → öncekine ekle
            chunks[-1] += " " + cur
        else:
            chunks.append(cur)
    return chunks


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def sections_to_chunks(sections: list[Section], source: str, max_chars=1200) -> list[dict]:
    chunks, counter = [], Counter()
    for s in sections:
        head = " > ".join([s.doc] + s.path)
        for part in split_text(s.text, max_chars):
            d = slug(s.doc)
            counter[d] += 1
            chunks.append({"id": f"{d}:{counter[d]}", "doc": s.doc, "group": s.group,
                           "section": " > ".join(s.path), "kind": s.kind, "source": source,
                           "pages": s.pages, "text": f"{head}\n{part}"})
    return chunks


if __name__ == "__main__":
    secs = parse_wiki_bundle(sys.argv[1])
    ch = sections_to_chunks(secs, "wikipedia")
    by = Counter(c["doc"] for c in ch)
    print(f"{len(secs)} bölüm, {len(ch)} chunk, {sum(len(c['text']) for c in ch):,} karakter")
    for d, n in by.most_common():
        print(f"  {d:40s} {n:4d}")
