"""Tüm ayarlar tek yerde. Değerler .env dosyasından okunur ve pydantic ile doğrulanır.

Notebook'ta ayarlar hücrelere dağılmıştı ve API anahtarı kodun içindeydi. Burada:
- anahtar sadece .env'de durur (.env asla git'e girmez),
- eksik veya yanlış tipte bir ayar varsa program AÇILIRKEN hata verir, ilk istekte değil.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent   # proje klasörü (dino/ klasörünün bir üstü)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8",
                                      extra="ignore")

    # ---- LLM ----
    llm_base_url: str = "your-model-or-deployment-name"
    llm_api_key: SecretStr                      # SecretStr: print edilince '**********' görünür
    llm_model: str = "gpt-4.1-mini"             # Azure'daki DEPLOYMENT adı
    # Fiyatı deployment adından tahmin etmiyoruz (notebook'taki usd=0.0 hatasının sebebi buydu:
    # deployment adı PRICES sözlüğünde yoktu). Fiyat açıkça ayar olarak verilir. $ / 1M token.
    price_in_per_m: float = 0.40
    price_out_per_m: float = 1.60

    # ---- Embedding ----
    embed_backend: Literal["local", "api"] = "local"
    emb_model_path: str = "models/multilingual-e5-small"   # local: klasör yolu
    emb_api_model: str = "text-embedding-3-small"          # api: deployment adı

    # ---- Dosyalar (göreli yollar proje klasörüne göre çözülür) ----
    sources_path: Path = Path("data/sources.json")     # hangi dokümanlar indekslenecek
    index_dir: Path = Path("data/index")
    species_csv: Path = Path("data/species.csv")
    tickets_path: Path = Path("data/tickets.jsonl")
    log_dir: Path = Path("logs")
    export_dir: Path = Path("exports")                 # kullanıcının istediği PDF/Word/Excel dosyaları

    # ---- Deneylerde kazanan ayarlar (Lab Bölüm 4-5) ----
    retrieval_k: int = Field(4, ge=1, le=10)
    rrf_k: int = Field(2, ge=1)
    use_rewrite: bool = True
    agent_max_steps: int = Field(6, ge=1, le=12)

    # ---- Deney bayrakları (eval bulgularından; .env'den aç/kapa, tek seferde tek değişken) ----
    search_spillover: bool = False      # A1: filtreli aramaya filtre dışından en iyi 2 chunk ekle
    answer_schema_v2: bool = False      # A3+A4: coverage/on_topic alanları, kaydı kod açar
    live_tools: bool = False            # A5: karşılaştırma araçları canlı Wikipedia + referans nesneler
    wiki_api_base: str = "https://en.wikipedia.org"   # testlerde sahte sunucuya yönlendirilir

    @model_validator(mode="after")
    def _resolve_paths(self):
        for name in ["sources_path", "index_dir", "species_csv", "tickets_path", "log_dir", "export_dir"]:
            p = getattr(self, name)
            if not p.is_absolute():
                setattr(self, name, ROOT / p)
        p = Path(self.emb_model_path)
        if self.embed_backend == "local" and not p.is_absolute():
            self.emb_model_path = str(ROOT / p)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
