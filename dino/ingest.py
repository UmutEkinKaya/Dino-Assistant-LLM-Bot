"""Kaynaklar → chunk'lar → embedding → diske kayıt.

Bir kez çalıştırılır (doküman eklenince/değişince tekrar):   python -m dino.ingest

Hangi dosyaların indeksleneceği data/sources.json'da yazar (manifest). Yeni doküman eklemek =
manifest'e bir satır. Her kaynağın kendi ayrıştırıcısı (parser) var; çıktıları aynı biçimde birleşir:
  {id, doc, group, section, kind, source, pages, text}     id örnekleri: guide:s4.1, tyrannosaurus:72

Notebook'ta indeks her açılışta baştan kuruluyordu. Burada indeks diske yazılır ve API açılırken
sadece okunur: açılış hızlanır, her açılışta aynı indeks kullanılır.

Neden hâlâ Chroma yok? ~1.300 chunk × 384 boyut = 2 MB'lık bir numpy matrisi; arama milisaniye.
Vektör veritabanı yüz binlerce chunk'ta, sürekli eklemede, çok süreçli erişimde gerekir.
"""
import json
import re
from datetime import datetime, timezone

import numpy as np
from pypdf import PdfReader

from .config import get_settings
from .embeddings import embedder_id, encode

HEADER_RE = re.compile(r"^Dinosaur Field Guide - Internal Training Edition\s*Rev\. 3\.1\s*$", re.M)
FOOTER_RE = re.compile(r"^Page \d+ of \d+\s+\|\s+For training use only\s*$", re.M)
HEAD_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?) ([A-Z][^\n]{2,60})$", re.M)


def clean_page(t: str) -> str:
    t = HEADER_RE.sub("", t)
    t = FOOTER_RE.sub("", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def load_text(pdf_path) -> str:
    pages = [p.extract_text() or "" for p in PdfReader(str(pdf_path)).pages]
    return "\n".join(clean_page(p) for p in pages)


def _parts(num: str) -> tuple[int, int | None]:
    a, _, b = num.partition(".")
    return int(a), (int(b) if b else None)


def sequential_heads(text: str) -> list[tuple[str, str, int]]:
    """Başlık adaylarını sadece SIRAYLA gelenler kabul edilecek şekilde süzer.

    "^12 Carnivore..." gibi bir tablo satırı da başlık regex'ine uyabilir (pypdf sürümüne ve
    tablonun metne nasıl döküldüğüne göre değişir). Gerçek başlıklar bir sıra izler:
    1 → 2 → 2.1 → 2.2 → 3 ... Bir önceki başlığın ardılı olmayan aday, gövde metninde kalır.
    """
    out, prev = [], None
    for m in HEAD_RE.finditer(text):
        num = m.group(1)
        cm, cs = _parts(num)
        if prev is None:
            ok = num == "1"
        else:
            pm, ps = prev
            ok = (cs is None and cm == pm + 1) or (cm == pm and cs == (ps or 0) + 1)
        if ok:
            out.append((num, m.group(2).strip(), m.start()))
            prev = (cm, cs)
    return out


def chunk_by_section(text: str) -> list[dict]:
    """Lab Bölüm 2'deki kazanan yöntem: her başlık bir chunk, alt bölümlere üst başlık eklenir."""
    heads = sequential_heads(text)
    titles = {num: title for num, title, _ in heads}
    chunks = []
    for i, (num, title, start) in enumerate(heads):
        end = heads[i + 1][2] if i + 1 < len(heads) else len(text)
        body = text[start:end].strip()
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.replace("\n", " ").strip()
        if not body:
            continue
        head = f"{num} {title}"
        parent = num.split(".")[0]
        if "." in num and parent in titles:
            head = f"{parent} {titles[parent]} > {head}"
        chunks.append({"id": f"s{num}", "section": num, "title": title, "text": f"{head}\n{body}"})
    return chunks


def load_guide(path, meta) -> list[dict]:
    """Kendi rehberimiz: numaralı başlıklarla bölünür (Lab Bölüm 2)."""
    text = load_text(path)
    n_cand = len(HEAD_RE.findall(text))
    out = []
    chunks = chunk_by_section(text)
    if len(chunks) < 10:
        raise SystemExit(f"{path.name} rehber gibi görünmüyor: sadece {len(chunks)} numaralı bölüm "
                         "bulundu (beklenen 38). Yanlış dosya mı? Rehber ile Wikipedia derlemesi "
                         "iki AYRI dosya olmalı.")
    if n_cand != len(chunks):
        print(f"  (rehber: {n_cand} başlık adayı, {len(chunks)} gerçek başlık; sırasız adaylar "
              "gövde metninde bırakıldı)")
    for c in chunks:
        out.append({"id": f"guide:{c['id']}", "doc": meta.get("title", "Field Guide"),
                    "group": "Rehber", "section": f"{c['section']} {c['title']}", "kind": "body",
                    "source": meta["source"], "pages": [], "text": c["text"]})
    return out


def load_wiki_bundle(path, meta) -> list[dict]:
    from .parsers import parse_wiki_bundle, sections_to_chunks
    return sections_to_chunks(parse_wiki_bundle(str(path)), meta["source"])


PARSERS = {"guide": load_guide, "wiki_bundle": load_wiki_bundle}


def load_manifest() -> list[dict]:
    s = get_settings()
    return json.loads(s.sources_path.read_text(encoding="utf-8"))


def main():
    s = get_settings()
    from .config import ROOT
    chunks = []
    for src in load_manifest():
        path = ROOT / src["path"]
        if not path.exists():
            raise SystemExit(f"Kaynak bulunamadı: {path} (data/sources.json'u kontrol et)")
        part = PARSERS[src["parser"]](path, src)
        print(f"  {src['path']:40s} {len(part):5d} chunk")
        chunks += part

    # Sağlık kontrolleri: format değişirse sessizce bozuk indeks kurmayalım
    ids = [c["id"] for c in chunks]
    if len(ids) != len(set(ids)):
        raise SystemExit("Aynı id'ye sahip chunk'lar var; parser'ı kontrol et.")
    leftovers = [c["id"] for c in chunks if re.search(r"doi:|ISBN|For training use only", c["text"])]
    if leftovers:
        print(f"UYARI: temizlenmemiş görünen {len(leftovers)} chunk: {leftovers[:5]}")

    print(f"{len(chunks)} chunk vektöre çevriliyor (CPU'da birkaç dakika sürebilir)...")
    E = encode([c["text"] for c in chunks], "passage").astype(np.float32)
    s.index_dir.mkdir(parents=True, exist_ok=True)
    np.save(s.index_dir / "embeddings.npy", E)
    (s.index_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False),
                                             encoding="utf-8")
    meta = {"embedder": embedder_id(), "n_chunks": len(chunks), "dim": int(E.shape[1]),
            "sources": [x["path"] for x in load_manifest()],
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (s.index_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{len(chunks)} chunk indekslendi → {s.index_dir}  (embedder: {meta['embedder']})")


if __name__ == "__main__":
    main()
