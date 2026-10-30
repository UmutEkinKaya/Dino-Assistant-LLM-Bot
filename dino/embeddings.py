"""Metni vektöre çeviren tek yer. İndeks kurarken (ingest) ve soru sorarken (retrieval)
AYNI fonksiyon kullanılır. İkisi farklı modelle yapılırsa vektörler karşılaştırılamaz;
bu yüzden kullanılan model kimliği indeksle birlikte kaydedilir ve açılışta kontrol edilir.
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import get_settings


def embedder_id() -> str:
    s = get_settings()
    if s.embed_backend == "local":
        return "local:" + Path(s.emb_model_path).name
    return "api:" + s.emb_api_model


@lru_cache
def _local_model():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # ofis ağı: internete çıkma
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import SentenceTransformer  # ağır import: sadece gerekince
    path = get_settings().emb_model_path
    if not Path(path).exists():
        raise FileNotFoundError(f"Embedding modeli bulunamadı: {path}. "
                                ".env içindeki EMB_MODEL_PATH ayarını kontrol et.")
    return SentenceTransformer(path, device="cpu")


def encode(texts: list[str], kind: str = "passage") -> np.ndarray:
    """kind: 'passage' (chunk) veya 'query' (soru). Normalize edilmiş matris döner."""
    s = get_settings()
    if s.embed_backend == "local":
        name = Path(s.emb_model_path).name.lower()
        if "e5" in name:                                   # e5 önek ister (Lab Deney 4.2)
            texts = [f"{kind}: {t}" for t in texts]
        return _local_model().encode(list(texts), normalize_embeddings=True, batch_size=32)

    from .llm import client                                # api backend
    rows = []
    for i in range(0, len(texts), 64):                     # API'nin istek başı sınırı var: parça parça
        r = client().embeddings.create(model=s.emb_api_model, input=list(texts[i:i + 64]))
        rows += [d.embedding for d in r.data]
    E = np.array(rows, dtype=np.float32)
    return E / np.linalg.norm(E, axis=1, keepdims=True)
