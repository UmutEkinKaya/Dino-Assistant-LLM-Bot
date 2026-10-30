"""/check akışı: iddia doğrulayıcı. Agent DEĞİL, sabit bir workflow:

  iddia → arama → (iddiada tür adı varsa) tablo satırları + zaman çizelgesi KODLA eklenir
        → FactCheck şeması → alıntılar KODLA doğrulanır

Neden agent değil? Adımlar her iddia için aynı. Sabit akış daha ucuz, daha hızlı ve daha
öngörülebilir. Agent'ı sadece adımların soruya göre değiştiği yerde kullan (/ask).
"""
import json
import re
import time

from . import llm
from .retrieval import normalize_source
from .schemas import FactCheck
from .tools import (CompareTimeline, ToolContext, _row, compare_timeline, species_in_text,
                    species_table)

SYSTEM = """You check claims against a dinosaur field guide. Use ONLY the sources below.
First copy short exact quotes as evidence, then decide the verdict.
If the guide says an older idea was revised, a claim repeating the older idea is 'refuted'.
If the guide says the question is open or studies disagree, the verdict is 'disputed'.
The claim may be in Turkish; quotes stay in the language of the sources."""


def _norm(t: str) -> str:
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t.lower())   # noktayı at (ondalık hariç: 0.02)
    t = re.sub(r"[^\w\s.-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def verify_quote(quote: str, source_text: str, threshold: float = 0.8) -> bool:
    """Alıntı kaynakta gerçekten geçiyor mu? Model alıntı UYDURABİLİR; bunu kod yakalar.

    Birebir geçiyorsa doğru. Değilse alıntının 3 kelimelik parçalarına bakılır: en az %80'i
    kaynakta geçiyorsa doğru sayılır (noktalama veya tek kelime farkı alıntıyı düşürmesin).
    """
    q, s = _norm(quote), _norm(source_text)
    if not q:
        return False
    if q in s:
        return True
    words = q.split()
    if len(words) < 3:
        return False
    grams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    return sum(g in s for g in grams) / len(grams) >= threshold


def build_sources(claim: str, retriever, k: int) -> tuple[dict[str, str], list[str]]:
    ids = retriever.search(claim, k)
    sources = {i: retriever.by_id[i]["text"] for i in ids}
    species = species_in_text(claim)
    if species:                                         # deterministik zenginleştirme
        names = {r["name"] for r in species}               # her türün iki kaynaktaki satırları
        tab = species_table()
        sources["table"] = json.dumps([_row(r) for _, r in tab[tab["name"].isin(names)].iterrows()],
                                      ensure_ascii=False)
    if len(species) >= 2:
        tl = compare_timeline(CompareTimeline(species_a=species[0]["name"],
                                              species_b=species[1]["name"]), ToolContext())
        if "summary" in tl:                             # yaş bilinmiyorsa araç hata döner: atla
            sources["timeline"] = tl["summary"]
    return sources, ids


def check(claim: str, retriever, k: int = 4) -> dict:
    t0 = time.time()
    sources, ids = build_sources(claim, retriever, k)
    context = "\n\n".join(f"[{sid}] {text}" for sid, text in sources.items())
    m = llm.parse([{"role": "system", "content": SYSTEM + "\n\nSources:\n" + context},
                   {"role": "user", "content": f"Claim: {claim}"}], FactCheck, tag="check")
    fc: FactCheck = m.parsed

    warnings, evidence = [], []
    for e in fc.evidence:
        sid = normalize_source(e.source)
        ok = sid in sources and verify_quote(e.quote, sources[sid])
        if sid not in sources:
            warnings.append(f"Bilinmeyen kaynak: {e.source}")
        elif not ok:
            warnings.append(f"Alıntı kaynakta bulunamadı ({sid}): {e.quote[:60]}")
        evidence.append({**e.model_dump(), "source": sid, "verified": ok})

    need = {"supported": "supports", "refuted": "refutes"}.get(fc.verdict)
    if need:
        reliable = any(e["verified"] and e["stance"] == need for e in evidence)
    elif fc.verdict == "disputed":
        reliable = any(e["verified"] for e in evidence)
    else:                                               # not_in_document: kanıt beklenmez
        reliable = True
    if not reliable:
        warnings.append("Karar, doğrulanmış bir alıntıyla desteklenmiyor: elle kontrol edin.")

    result = {**fc.model_dump(), "evidence": evidence}
    return {"result": result, "reliable": reliable, "retrieved": list(sources),
            "warnings": warnings, "latency_ms": int((time.time() - t0) * 1000)}
