"""Regresyon testi: kodda, prompt'ta veya ayarda her değişiklikten sonra çalıştır.

  python -m dino.eval retrieval          # sadece arama (ucuz)
  python -m dino.eval ask   [--limit 5]  # agent: cevap kalitesi + araç kullanımı
  python -m dino.eval check              # iddia doğrulayıcı: karar doğruluğu
  python -m dino.eval all

Her koşu reports/ altına satır satır CSV yazar ve özetini reports/history.jsonl'e ekler.
Bir önceki koşuyla fark otomatik basılır: "değişikliğim neyi iyileştirdi, neyi bozdu?"
"""
import argparse
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from . import agent, factcheck, llm
from .config import ROOT
from .retrieval import Retriever

DATA, REPORTS = ROOT / "data", ROOT / "reports"


# ------------------------------------------------------------------------------------------
# Yargıçlar (Lab eval3'teki düzeltilmiş hali): gerekçe ve dayanaksız iddialar SKORDAN ÖNCE
# ------------------------------------------------------------------------------------------
class FaithJudgement(BaseModel):
    unsupported_claims: list[str] = Field(description="Every claim in the answer that is NOT "
                                          "stated in the context, even if true in the real world")
    reason: str = Field(description="Short justification, written BEFORE the score")
    score: float = Field(description="0 to 1")


class RefusalJudgement(BaseModel):
    reason: str = Field(description="Short justification, written BEFORE the decision")
    refused: bool
    invented: bool


class CorrJudgement(BaseModel):
    reason: str = Field(description="Short justification, written BEFORE the score")
    score: float = Field(description="0 to 1")


# Yargıç v2 (ilk eval'in kalibrasyonundan): yargıç (1) sorunun öncülünü cevabın iddiası sanıyordu,
# (2) "Geç Kretase'de yaşadı → Jura'da yaşamadı" gibi doğrudan çıkarımı dayanaksız sayıyordu,
# (3) araç çağrısının parametrelerini görmüyordu ("otoburları kütleye göre artan sırala, 1 tane").
FAITH = ("You check GROUNDING, not truth. A claim is supported if the context states it OR it "
         "follows directly from the context by simple logic (e.g. 'lived in the Late Cretaceous' "
         "supports 'did not live in the Jurassic'; a tool call that sorts by mass ascending with "
         "limit 1 supports 'the lightest'). A claim true in the real world but absent from the "
         "context is UNSUPPORTED. Restating the user's premise in order to deny or correct it is "
         "NOT a claim; judge the correction instead. Saying that the context lacks some "
         "information is supported. Score = share of supported claims.")
REFUSAL = ("The question cannot be answered from the documents (or is off-topic). Judge the answer: "
           "'refused' = it clearly says the documents do not contain the answer (or that the "
           "assistant only covers dinosaurs). 'invented' = it nevertheless presents an answer to the "
           "main question as if it came from the documents (e.g. a guess written as a document fact "
           "or inference). Related true facts from the documents are fine.")
CORR = ("Compare the answer with the expected answer. If a list of MUST-HAVE facts is given, the "
        "answer needs all of them for 1; other details in the expected answer are nice-to-have. "
        "1 = all must-have facts present, 0.5 = some missing, 0 = wrong or contradicts the expected answer.")
OUT_MARK = re.compile(r"(Outside the document|Belge dışı bilgi|Based on general knowledge)[,:]?.*",
                      re.S | re.I)


def judge(rule, schema, q, answer, context="", expected=""):
    m = llm.parse([{"role": "system", "content": "You are a strict evaluator. " + rule},
                   {"role": "user", "content": f"Question: {q}\n\nContext:\n{context}\n\n"
                                               f"Expected answer: {expected}\n\nAnswer: {answer}"}],
                  schema, tag="judge")
    return m.parsed


def grounded_text(ans) -> str:
    """Yargıca sadece belgeye dayanan katmanlar gider; genel bilgi gitmez."""
    parts = [f"Correction: {t}" for t in ans.corrections]
    if getattr(ans, "document_says", ""):
        parts.append(f"What the documents say: {ans.document_says}")
    parts += [f"Document fact: {c.text} {c.sources}" for c in ans.document]
    parts += [f"Inference: {c.text} {c.sources}" for c in ans.inference]
    parts += [f"Web fact (live Wikipedia / reference): {c.text} {c.sources}"
              for c in getattr(ans, "web_facts", [])]                      # v2 + canlı araçlar
    parts += [f"Debated: {t}" for t in getattr(ans, "debated", [])]        # v2 şeması
    summ = OUT_MARK.sub("", ans.summary).strip()
    if summ:
        parts.append(f"Summary: {summ}")
    return "\n".join(parts) if parts else "(no document-based content)"


def is_gold(chunk_id: str, g: dict, ret: Retriever) -> bool:
    """Doğru kaynak mı? Rehberde bölüm düzeyinde (guide:s4.1), Wikipedia'da makale düzeyinde.
    Büyük korpusta chunk düzeyinde gold işaretlemek pahalı; makale düzeyi pratik bir orta yol."""
    # "4" altındaki "4.1", "4.2"... de doğru sayılır (bölüm ve alt bölümleri)
    if any(chunk_id == f"guide:s{x}" or chunk_id.startswith(f"guide:s{x}.") for x in g["gold"]):
        return True
    c = ret.by_id.get(chunk_id) if ret is not None else None
    return bool(c and c["doc"] in g.get("gold_docs", []))


def evidence_ids(out: dict) -> list[str]:
    """Agent'ın gördüğü TÜM kaynaklar: aramadan gelen chunk'lar + araç çıktılarındaki kaynaklar
    (ör. compare_timeline'ın döndürdüğü guide:s4.1). Sadece aramaya bakmak, tabloyla doğru
    cevap veren soruları 'yanlış kaynak' sayıyordu."""
    from .retrieval import SOURCE_ID_RE
    ids = list(out["seen_chunks"])
    for t in out["tool_outputs"]:
        ids += [i for i in SOURCE_ID_RE.findall(t) if i not in ids]
    return ids


def tools_ok(expected: list, used: list[str]) -> float:
    """expected: ["compare_size"] = bu araç şart; [["find_species", "compare_timeline"]] = biri yeter.
    Aynı soruya giden birden çok doğru yol olabilir; tek bir yolu şart koşmak doğru cevabı cezalandırır."""
    ok = all((any(t in used for t in e) if isinstance(e, list) else e in used) for e in expected)
    return float(ok)


def load_jsonl(name):
    return [json.loads(l) for l in (DATA / name).read_text(encoding="utf-8").splitlines() if l.strip()]


# ------------------------------------------------------------------------------------------
# 1) Retrieval
# ------------------------------------------------------------------------------------------
def eval_retrieval(ret: Retriever, golden, limit=None):
    rows = []
    for g in golden[:limit]:
        if not g["gold"] and not g.get("gold_docs"):
            continue
        ids = ret.search(g["question"], 5)
        ranks = [i for i, d in enumerate(ids) if is_gold(d, g, ret)]
        rows.append({"id": g["id"], "category": g["category"], "ids": ids,
                     "hit@1": float(bool(ranks) and ranks[0] < 1),
                     "hit@3": float(bool(ranks) and ranks[0] < 3),
                     "mrr": 1 / (ranks[0] + 1) if ranks else 0.0})
        print(".", end="", flush=True)
    df = pd.DataFrame(rows)
    return df, df[["hit@1", "hit@3", "mrr"]].mean().round(3).to_dict()


# ------------------------------------------------------------------------------------------
# 2) Ask (agent): cevap kalitesi + araç davranışı
# ------------------------------------------------------------------------------------------
def eval_ask(ret: Retriever, golden, limit=None):
    tickets = REPORTS / "eval_tickets.jsonl"          # eval gerçek kayıt dosyasını kirletmesin
    tickets.unlink(missing_ok=True)
    rows = []
    for g in golden[:limit]:
        r = {"id": g["id"], "category": g["category"], "question": g["question"],
             "expected": g["expected"], "must": " | ".join(g.get("must", []))}
        special = g["expected"] in ("NOT_IN_DOCS", "REFUSE_OR_REDIRECT")
        r["ticket_expected"] = float(g["expected"] == "NOT_IN_DOCS")
        try:
            with llm.meter() as m:
                out = agent.ask(g["question"], ret, tickets_path=tickets)
                ans = out["answer"]
                has_gold = bool(g["gold"] or g.get("gold_docs"))
                r.update(steps=out["steps"], tools=",".join(out["tools_used"]),
                         ticket_opened=float(bool(out["tickets_opened"])),
                         warnings=len(out["warnings"]), summary=ans.summary,
                         # okunabilir rapor için cevabın tamamı (python -m dino.report)
                         grounded=grounded_text(ans),
                         general_text=" | ".join(getattr(g, "text", g) for g in ans.general_knowledge),
                         not_in_doc=" | ".join(ans.not_in_document),
                         tickets=",".join(out["tickets_opened"]),
                         trace=" → ".join(f"{t['tool']}({json.dumps({k: v for k, v in t['args'].items() if v is not None}, ensure_ascii=False)})"
                                          for t in out["trace"]),
                         seen=",".join(evidence_ids(out)),
                         web_used=float(any(i.startswith(("web:", "ref:")) for i in evidence_ids(out))),
                         warning_text=" | ".join(out["warnings"]),
                         general_knowledge=float(bool(ans.general_knowledge)),
                         correct_chunk=float(any(is_gold(c, g, ret) for c in evidence_ids(out)))
                         if has_gold else np.nan,
                         tool_ok=tools_ok(g["expected_tools"], out["tools_used"])
                         if g["expected_tools"] else np.nan)
                if hasattr(ans, "coverage"):
                    r.update(coverage=ans.coverage, on_topic=float(ans.on_topic))
                if special:
                    # Ret, "belge katmanı boş mu?" ile ölçülmez: model ilgili gerçekleri yazabilir.
                    # Doğru ret = "belgede yok" dedi VE uydurma bir cevabı belge gibi sunmadı.
                    rj = judge(REFUSAL, RefusalJudgement, g["question"],
                               grounded_text(ans) + f"\nNot in document: {ans.not_in_document}")
                    r.update(correct_refusal=float(rj.refused and not rj.invented), f_reason=rj.reason)
                else:
                    gt = grounded_text(ans)
                    calls = "\n".join(f"TOOL CALL: {t['tool']}({json.dumps(t['args'], ensure_ascii=False)})"
                                      for t in out["trace"])
                    # Kısa hesap/tablo/canlı sonuçlar önce, uzun belge aramaları sonra: 40k kesimi
                    # canlı Wikipedia sonucunu yutmasın (L1'de hakem onu göremeyip 'dayanaksız' demişti).
                    outs = sorted(out["tool_outputs"], key=lambda t: t.startswith("[search_docs]"))
                    ctx = (calls + "\n\nTOOL RESULTS:\n" + "\n\n".join(outs))[:40000]
                    f = judge(FAITH, FaithJudgement, g["question"], gt, context=ctx)
                    exp = g["expected"] + (f"\nMUST-HAVE facts: {'; '.join(g['must'])}"
                                           if g.get("must") else "")
                    c = judge(CORR, CorrJudgement, g["question"], gt, expected=exp)
                    r.update(faithfulness=f.score, correctness=c.score,
                             unsupported=" | ".join(f.unsupported_claims), c_reason=c.reason,
                             f_reason=f.reason)
            r["usd"] = m["usd"]
        except llm.ContentFiltered:
            r["filtered"] = 1.0
            r["correct_refusal"] = 1.0 if special else 0.0
        rows.append(r)
        print(".", end="", flush=True)
    df = pd.DataFrame(rows)
    cols = [c for c in ["correctness", "correct_refusal", "tool_ok", "ticket_opened"] if c in df]
    print("\n\nKategori bazında:")
    tab = df.groupby("category")[cols].mean().round(2)
    tab.insert(0, "n", df.groupby("category").size())          # az sorulu kategori = gürültülü
    print(tab.to_string())
    t = df.dropna(subset=["ticket_opened"])
    tp = ((t.ticket_opened == 1) & (t.ticket_expected == 1)).sum()
    summary = {k: round(float(df[k].mean()), 3) for k in
               ["correctness", "faithfulness", "correct_refusal", "correct_chunk", "tool_ok",
                "general_knowledge", "web_used", "steps"] if k in df}
    summary["usd_per_question"] = round(float(df["usd"].mean()), 5) if "usd" in df else 0.0
    opened, expected = t.ticket_opened.sum(), t.ticket_expected.sum()
    # Hiç kayıt açılmadıysa precision tanımsızdır (0 değil): None yazıyoruz
    summary["ticket_precision"] = round(tp / opened, 3) if opened else None
    summary["ticket_recall"] = round(tp / expected, 3) if expected else None
    summary["usd_total"] = round(float(df.get("usd", pd.Series([0])).sum()), 4)
    return df, summary


# ------------------------------------------------------------------------------------------
# 3) Check (fact-check): bir sınıflandırma problemi → confusion matrix
# ------------------------------------------------------------------------------------------
def eval_check(ret: Retriever, claims, limit=None):
    rows = []
    for c in claims[:limit]:
        r = {"id": c["id"], "claim": c["claim"], "label": c["label"]}
        try:
            out = factcheck.check(c["claim"], ret)
            ev = out["result"]["evidence"]
            res = out["result"]
            r.update(explanation=res["explanation"], correction=res.get("correction") or "",
                     evidence=" | ".join(f"[{e['source']}{' ✓' if e['verified'] else ' ✗'} "
                                         f"{e['stance']}] {e['quote']}" for e in ev),
                     retrieved=",".join(out["retrieved"]), warning_text=" | ".join(out["warnings"]))
            r.update(pred=out["result"]["verdict"], reliable=float(out["reliable"]),
                     verified_quotes=float(np.mean([e["verified"] for e in ev])) if ev else np.nan)
        except llm.ContentFiltered:
            r["pred"] = "filtered"
        rows.append(r)
        print(".", end="", flush=True)
    df = pd.DataFrame(rows)
    df["correct"] = (df.label == df.pred).astype(float)
    print("\n\nConfusion matrix (satır = gerçek, sütun = tahmin):")
    print(pd.crosstab(df.label, df.pred))
    print("\nYanlışlar:")
    for r in df[df.correct == 0].itertuples():
        print(f"  {r.id} [{r.label} → {r.pred}] {r.claim}")
    per_class = df.groupby("label")["correct"].mean().round(3).to_dict()
    summary = {"accuracy": round(float(df.correct.mean()), 3),
               "reliable": round(float(df.reliable.mean()), 3) if "reliable" in df else None,
               "verified_quotes": round(float(df.verified_quotes.mean()), 3)
               if "verified_quotes" in df else None,
               **{f"recall_{k}": v for k, v in per_class.items()}}
    return df, summary


# ------------------------------------------------------------------------------------------
def save_and_compare(kind: str, df: pd.DataFrame, summary: dict, note: str):
    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(REPORTS / f"{stamp}_{kind}.csv", index=False, encoding="utf-8-sig")
    hist = REPORTS / "history.jsonl"
    prev = None
    if hist.exists():
        for line in hist.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["kind"] == kind and row["n"] == len(df):   # sadece aynı soru setiyle kıyasla
                prev = row
    with open(hist, "a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": kind, "time": stamp, "note": note, "n": len(df),
                            **summary}, ensure_ascii=False) + "\n")
    print(f"\n\n== {kind}  (n={len(df)})  → reports/{stamp}_{kind}.csv")
    if kind in ("ask", "check", "live"):
        from .report import write_report
        print(f"   okunabilir rapor → {write_report(REPORTS / f'{stamp}_{kind}.csv').name}"
              "  (tarayıcıda aç)")
    for k, v in summary.items():
        line = f"  {k:18s} {v}"
        if prev and isinstance(v, (int, float)) and isinstance(prev.get(k), (int, float)):
            d = v - prev[k]
            if abs(d) > 1e-9:
                line += f"   ({'+' if d > 0 else ''}{d:.3f} önceki koşuya göre: {prev['note'] or prev['time']})"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["retrieval", "ask", "check", "live", "all"])
    ap.add_argument("--limit", type=int, default=None, help="ilk N soru (hızlı deneme için)")
    ap.add_argument("--note", default="", help="bu koşuda neyi değiştirdin? ör. 'prompt v2'")
    a = ap.parse_args()
    ret = Retriever.load()
    golden, claims = load_jsonl("golden_set.jsonl"), load_jsonl("claims.jsonl")
    kinds = ["retrieval", "ask", "check"] if a.kind == "all" else [a.kind]
    for k in kinds:
        fn = {"retrieval": lambda: eval_retrieval(ret, golden, a.limit),
              "ask": lambda: eval_ask(ret, golden, a.limit),
              "live": lambda: eval_ask(ret, load_jsonl("golden_live.jsonl"), a.limit),
              "check": lambda: eval_check(ret, claims, a.limit)}[k]
        df, summary = fn()
        save_and_compare(k, df, summary, a.note)


if __name__ == "__main__":
    main()
