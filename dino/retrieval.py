"""Arama: Lab'daki deneylerin kazananı sabitlendi.

  e5 dense  +  LLM ile İngilizce query rewrite  +  RRF (k=2)
  Türkçe soruda orijinal soru RRF'e KATILMAZ (Deney 5.3: Türkçe sorgu sıralamayı bozuyordu).

BM25/hybrid yok: bu dokümanda dense'ten kötü çıktı (Deney 4.x). Başka dokümana geçersen tekrar ölç.
"""
import json
import re
from functools import lru_cache

import numpy as np
from pydantic import BaseModel, Field

from . import llm
from .config import get_settings
from .embeddings import embedder_id, encode


class IndexNotReady(RuntimeError):
    pass


class Rewrite(BaseModel):
    queries: list[str] = Field(description="2-3 English search queries using scientific vocabulary")


SOURCE_ID_RE = re.compile(r"\b(?:guide:s\d{1,2}(?:\.\d{1,2})?|[a-z0-9]+(?:-[a-z0-9]+)*:\d+"
                          r"|web:[A-Z][A-Za-z]+|ref:[a-z_]+)\b")   # web: canlı Wikipedia, ref: referans nesne


def normalize_source(s: str) -> str:
    """Modelin yazdığı kaynak id'sini standart biçime getirir: '4.2', 's4.2' → 'guide:s4.2'."""
    s = s.strip().strip("[]").strip()
    if re.fullmatch(r"s?\d{1,2}(?:\.\d{1,2})?", s):
        return "guide:" + (s if s.startswith("s") else "s" + s)
    return s


def is_turkish(q: str) -> bool:
    return bool(re.search(r"[çğıöşüÇĞİÖŞÜ]", q))


def rrf(rankings: list[list[str]], k: int) -> list[str]:
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking):
            score[d] = score.get(d, 0) + 1 / (k + rank + 1)
    return sorted(score, key=score.get, reverse=True)


@lru_cache(maxsize=512)
def rewrite(q: str) -> tuple[str, ...]:
    m = llm.parse([{"role": "system", "content": "Rewrite the user's question (any language) into "
                    "English search queries using the vocabulary a paleontology text would use."},
                   {"role": "user", "content": q}], Rewrite, tag="rewrite")
    return tuple(m.parsed.queries[:3])


class Retriever:
    def __init__(self, chunks: list[dict], E: np.ndarray):
        self.chunks = chunks
        self.by_id = {c["id"]: c for c in chunks}
        self.ids = [c["id"] for c in chunks]
        self.E = E
        self.src = np.array([c.get("source", "") for c in chunks])
        self.doc = np.array([c.get("doc", "") for c in chunks])
        self.articles = sorted({c["doc"] for c in chunks if c.get("source") == "wikipedia"})

    @classmethod
    def load(cls, index_dir=None) -> "Retriever":
        d = index_dir or get_settings().index_dir
        if not (d / "meta.json").exists():
            raise IndexNotReady(f"İndeks bulunamadı: {d}. Önce `python -m dino.ingest` çalıştır.")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta["embedder"] != embedder_id():
            raise IndexNotReady(f"İndeks '{meta['embedder']}' ile kurulmuş, şu anki ayar "
                              f"'{embedder_id()}'. Aynı model olmalı: `python -m dino.ingest` "
                              "ile indeksi yeniden kur.")
        chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        return cls(chunks, np.load(d / "embeddings.npy"))

    def dense(self, q: str, n: int = 20, mask=None) -> list[str]:
        scores = self.E @ encode([q], "query")[0]
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)        # filtre dışı chunk'lar asla gelmez
        order = [i for i in np.argsort(-scores)[:n] if np.isfinite(scores[i])]
        return [self.ids[i] for i in order]

    def queries_for(self, q: str, rewrite_mode: str = "always") -> list[str]:
        """rewrite_mode: 'always' = ham kullanıcı sorusu (fact-check, retrieval eval): LLM ile yeniden yaz.
        'auto' = agent'ın yazdığı sorgu: zaten İngilizce ve terimli, sadece Türkçe kaçtıysa yeniden yaz.
        (Lab'dan kalma: agent'ta her aramada ek bir LLM çağrısı = ~1-1.5 sn boşa gidiyordu.)"""
        if not get_settings().use_rewrite or (rewrite_mode == "auto" and not is_turkish(q)):
            return [q]
        rw = list(rewrite(q))
        return rw if is_turkish(q) else [q] + rw

    def search(self, q: str, k: int | None = None, source: str | None = None,
               doc: str | None = None, rewrite_mode: str = "always") -> list[str]:
        """source / doc: metadata filtresi (ör. sadece rehber, sadece 'Tyrannosaurus' makalesi)."""
        s = get_settings()
        k = k or s.retrieval_k
        mask = None
        if source or doc:
            mask = np.ones(len(self.ids), dtype=bool)
            if source:
                mask &= self.src == source
            if doc:
                mask &= self.doc == doc
            if not mask.any():
                return []
        return rrf([self.dense(x, 20, mask) for x in self.queries_for(q, rewrite_mode)], s.rrf_k)[:k]

    def format(self, ids: list[str]) -> str:
        def head(i):
            p = self.by_id[i].get("pages") or []
            return f"[{i}]" + (f" (pp. {p[0]}-{p[-1]})" if p else "")
        return "\n\n".join(f"{head(i)} {self.by_id[i]['text']}" for i in ids)
