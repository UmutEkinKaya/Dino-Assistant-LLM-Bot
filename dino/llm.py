"""LLM'e giden TEK kapı. Her çağrı buradan geçer: süre, token ve maliyet loglanır,
sağlayıcı hataları uygulamanın anlayacağı hata türlerine çevrilir.

Neden tek kapı? Model değiştirmek, timeout eklemek, maliyet ölçmek gibi işler tek dosyada yapılır.
"""
import json
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import lru_cache

from openai import (APIConnectionError, APITimeoutError, BadRequestError, OpenAI,
                    RateLimitError, LengthFinishReasonError,
                    ContentFilterFinishReasonError)
from pydantic import ValidationError

from .config import get_settings


# ------------------------------------------------------------------------------------------
# Hata türleri. API katmanı bunları HTTP durum kodlarına çevirir (api.py).
# ------------------------------------------------------------------------------------------
class LLMError(Exception):
    status = 502
    code = "llm_error"


class ContentFiltered(LLMError):        # Azure içerik filtresi (lab'da q27'de yaşadık)
    status = 400
    code = "content_filtered"


class LLMUnavailable(LLMError):         # timeout / bağlantı yok
    status = 503
    code = "llm_unavailable"


class LLMRateLimited(LLMError):         # kota doldu
    status = 429
    code = "llm_rate_limited"


class SchemaError(LLMError):            # model şemaya uygun cevap üretemedi
    status = 502
    code = "schema_error"


# ------------------------------------------------------------------------------------------
# Maliyet sayacı: bir isteğin (ör. tek bir /ask) toplam çağrı, token ve $ değerini toplar.
# ContextVar: aynı anda gelen iki isteğin sayaçları birbirine karışmaz.
# ------------------------------------------------------------------------------------------
_meter: ContextVar = ContextVar("meter", default=None)
_log_lock = threading.Lock()


@contextmanager
def meter():
    m = {"calls": 0, "in_tok": 0, "out_tok": 0, "usd": 0.0}
    token = _meter.set(m)
    try:
        yield m
    finally:
        _meter.reset(token)


@lru_cache
def client() -> OpenAI:
    s = get_settings()
    return OpenAI(
        base_url=s.llm_base_url,                       # varsayılan: (config.py)
        api_key=s.llm_api_key.get_secret_value(),      # anahtar .env'deki LLM_API_KEY satırından
        timeout=60, max_retries=2,                     # 429/5xx'te SDK kendisi 2 kez dener
    )


def _record(tag: str, t0: float, usage) -> None:
    s = get_settings()
    i = getattr(usage, "prompt_tokens", 0) or 0
    o = getattr(usage, "completion_tokens", 0) or 0
    usd = (i * s.price_in_per_m + o * s.price_out_per_m) / 1e6
    m = _meter.get()
    if m is not None:
        with _log_lock:                                  # tablo aracı aramaları paralel yapar
            m["calls"] += 1; m["in_tok"] += i; m["out_tok"] += o; m["usd"] += usd
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tag": tag,
           "model": s.llm_model, "sec": round(time.time() - t0, 2), "in_tok": i, "out_tok": o,
           "usd": round(usd, 6)}
    s.log_dir.mkdir(parents=True, exist_ok=True)
    with _log_lock, open(s.log_dir / "calls.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_content_filter(e: Exception) -> bool:
    text = str(e).lower()
    return "content_filter" in text or "responsibleaipolicyviolation" in text


def _call(fn, tag: str, **kw):
    t0 = time.time()
    try:
        r = fn(model=get_settings().llm_model, **kw)
    except ContentFilterFinishReasonError as e:            # parse() cevap filtresini böyle bildirir
        raise ContentFiltered("Cevap içerik filtresine takıldı.") from e
    except BadRequestError as e:
        if _is_content_filter(e):
            raise ContentFiltered("İstek içerik filtresine takıldı.") from e
        raise LLMError(f"LLM isteği reddetti: {e.message}") from e
    except RateLimitError as e:
        raise LLMRateLimited("LLM kotası doldu, biraz sonra tekrar deneyin.") from e
    except (APITimeoutError, APIConnectionError) as e:
        raise LLMUnavailable("LLM'e ulaşılamadı (timeout veya bağlantı).") from e
    _record(tag, t0, r.usage)
    if r.choices[0].finish_reason == "content_filter":      # cevap tarafında filtre
        raise ContentFiltered("Cevap içerik filtresine takıldı.")
    return r


def chat(messages, tag="chat", temperature=0, max_tokens=800, **kw):
    """Serbest metin (veya tool çağrısı) dönen çağrı. Mesaj nesnesini döndürür."""
    r = _call(client().chat.completions.create, tag, messages=messages,
              temperature=temperature, max_completion_tokens=max_tokens, **kw)
    return r.choices[0].message


def parse(messages, schema, tag="parse", temperature=0, retries=1, **kw):
    """Structured output: cevap pydantic şemasına uymak zorunda.

    Tool'larla birlikte de kullanılabilir (agent.py): o zaman model ya tool çağırır
    (message.tool_calls dolu, parsed None) ya da şemaya uygun son cevabı verir.
    Mesaj nesnesini döndürür; şemalı sonuç message.parsed içindedir.
    """
    last = None
    for attempt in range(retries + 1):
        try:
            r = _call(client().chat.completions.parse, f"{tag}" if attempt == 0 else f"{tag}:retry",
                      messages=messages, response_format=schema, temperature=temperature, **kw)
        except (LengthFinishReasonError, ValidationError) as e:   # yarım kalan / bozuk JSON
            last = e
            continue
        m = r.choices[0].message
        if m.parsed is not None or m.tool_calls:
            return m
        last = m.refusal or "boş cevap"
    raise SchemaError(f"Model şemaya uygun cevap üretemedi: {str(last)[:200]}")
