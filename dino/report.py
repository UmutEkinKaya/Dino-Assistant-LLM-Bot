"""Eval CSV'sini okunabilir hale getirir: tarayıcıda açılan tek bir HTML dosyası.

  python -m dino.report                 # en son ask raporu
  python -m dino.report reports/20261026_154821_ask.csv
  python -m dino.report --top 10        # ayrıca en kötü 10 soruyu terminale düz metin bas

CSV makine için (pandas, kıyaslama), HTML insan için: her soru bir kart, en kötüler en üstte,
kategori filtresi, "sadece sorunlular" kutusu. Eski CSV'lerde olmayan sütunlar atlanır.
"""
import argparse
import html
import math
from pathlib import Path

import pandas as pd

from .config import ROOT

REPORTS = ROOT / "reports"
def _txt(x) -> str:
    """NaN / None / 'nan' → boş metin (boş CSV hücresi pandas'ta NaN olarak okunur)."""
    if x is None or (isinstance(x, float) and math.isnan(x)) or str(x).strip().lower() == "nan":
        return ""
    return str(x)


E = lambda x: html.escape(_txt(x))


def _num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _badge(label, v, good=1.0):
    f = _num(v)
    if f is None:
        return ""
    cls = "ok" if f >= good else ("mid" if f >= 0.5 else "bad")
    return f'<span class="b {cls}">{label} {f:.2g}</span>'


def _problem_ask(r) -> bool:
    checks = [_num(r.get("correctness")) is not None and _num(r.get("correctness")) < 1,
              _num(r.get("correct_refusal")) == 0, _num(r.get("tool_ok")) == 0,
              _num(r.get("ticket_opened")) is not None
              and _num(r.get("ticket_opened")) != _num(r.get("ticket_expected"))]
    return any(checks)


def _score_ask(r) -> float:
    vals = [_num(r.get(k)) for k in ["correctness", "correct_refusal", "tool_ok"]]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else 1.0


def _lines(text, cls="") -> str:
    if not isinstance(text, str):                     # NaN / boş hücre
        return ""
    parts = [p for p in text.split("\n") if p.strip()]
    return "".join(f'<div class="{cls}">{E(p)}</div>' for p in parts)


def _field(title, body):
    return f'<div class="f"><div class="ft">{title}</div>{body}</div>' if body else ""


def card_ask(r) -> str:
    badges = (_badge("correctness", r.get("correctness")) + _badge("faith", r.get("faithfulness"))
              + _badge("ret", r.get("correct_refusal")) + _badge("araç", r.get("tool_ok"))
              + _badge("doğru kaynak", r.get("correct_chunk")))
    t_exp, t_open = _num(r.get("ticket_expected")), _num(r.get("ticket_opened"))
    if t_open is not None and t_exp is not None and t_open != t_exp:
        badges += f'<span class="b bad">kayıt: {"açtı" if t_open else "açmadı"} (beklenen tersi)</span>'
    unsupported = "".join(f"<li>{E(u)}</li>" for u in str(r.get("unsupported") or "").split(" | ")
                          if u.strip() and u != "nan")
    body = (
        _field("Beklenen", f'<div>{E(r.get("expected"))}</div>'
               + (f'<div class="must">olmazsa olmaz: {E(r.get("must"))}</div>'
                  if r.get("must") and r.get("must") == r.get("must") else ""))
        + _field("Cevap (belgeye dayanan katmanlar)", _lines(r.get("grounded") or r.get("summary") or ""))
        + _field("Genel bilgi (belge dışı, puanlanmaz)", _lines(_txt(r.get("general_text")).replace(" | ", "\n"), "gk"))
        + _field("Belgede yok dedikleri", _lines(_txt(r.get("not_in_doc")).replace(" | ", "\n")))
        + _field("Araç adımları", f'<div class="mono">{E(r.get("trace") or r.get("tools"))}</div>')
        + _field("Kayıt", E(r.get("tickets")))
        + _field("Yargıç: correctness gerekçesi", E(r.get("c_reason")))
        + _field("Yargıç: dayanaksız iddialar", f"<ul>{unsupported}</ul>" if unsupported else "")
        + _field("Uyarılar", E(r.get("warning_text")))
    )
    return (f'<details class="card" data-cat="{E(r.get("category"))}" '
            f'data-bad="{int(_problem_ask(r))}"{" open" if _problem_ask(r) else ""}>'
            f'<summary><b>{E(r.get("id"))}</b> <span class="cat">{E(r.get("category"))}</span> '
            f'{E(r.get("question"))}<div class="badges">{badges}</div></summary>{body}</details>')


def card_check(r) -> str:
    ok = r.get("label") == r.get("pred")
    ev = "".join(f'<li class="mono">{E(e)}</li>' for e in str(r.get("evidence") or "").split(" | ")
                 if e.strip() and e != "nan")
    body = (_field("Açıklama", E(r.get("explanation"))) + _field("Düzeltme", E(r.get("correction")))
            + _field("Kanıtlar (✓ = alıntı kaynakta bulundu)", f"<ul>{ev}</ul>" if ev else "")
            + _field("Getirilen kaynaklar", f'<div class="mono">{E(r.get("retrieved"))}</div>')
            + _field("Uyarılar", E(r.get("warning_text"))))
    verdict = (f'<span class="b {"ok" if ok else "bad"}">etiket: {E(r.get("label"))} → '
               f'tahmin: {E(r.get("pred"))}</span>')
    return (f'<details class="card" data-cat="{E(r.get("label"))}" data-bad="{int(not ok)}"'
            f'{"" if ok else " open"}><summary><b>{E(r.get("id"))}</b> {E(r.get("claim"))}'
            f'<div class="badges">{verdict}{_badge("güvenilir", r.get("reliable"))}</div></summary>'
            f"{body}</details>")


CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--card:#fff;--line:#e7e5e4;--ok:#15803d;--mid:#b45309;--bad:#b91c1c;--gk:#f5f5f4}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--card:#292524;--line:#44403c;--ok:#4ade80;--mid:#fbbf24;--bad:#f87171;--gk:#3a3532}}
body{font:14px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:1000px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:4px 0}.mut{color:var(--mut)}table{border-collapse:collapse;margin:8px 0}
td,th{border:1px solid var(--line);padding:3px 8px;text-align:right}td:first-child,th:first-child{text-align:left}
.bar{position:sticky;top:0;background:var(--bg);padding:8px 0;display:flex;gap:12px;flex-wrap:wrap;border-bottom:1px solid var(--line)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:8px 12px}
summary{cursor:pointer}.cat{color:var(--mut);font-size:12px}.badges{margin-top:4px}
.b{display:inline-block;font-size:12px;border:1px solid currentColor;border-radius:10px;padding:0 7px;margin:2px 4px 0 0}
.ok{color:var(--ok)}.mid{color:var(--mid)}.bad{color:var(--bad)}
.f{margin:8px 0}.ft{font-size:12px;font-weight:600;color:var(--mut);text-transform:uppercase;letter-spacing:.03em}
.gk{background:var(--gk);padding:2px 6px;border-radius:4px;margin:2px 0}.must{color:var(--mid);font-size:13px}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:12px;word-break:break-word}ul{margin:4px 0;padding-left:20px}
"""

JS = """
const sel=document.getElementById('cat'),bad=document.getElementById('bad'),q=document.getElementById('q');
function f(){document.querySelectorAll('.card').forEach(c=>{const t=c.textContent.toLowerCase();
c.style.display=((!sel.value||c.dataset.cat===sel.value)&&(!bad.checked||c.dataset.bad==='1')
&&(!q.value||t.includes(q.value.toLowerCase())))?'':'none'})}
[sel,bad,q].forEach(e=>e.addEventListener('input',f));
"""


def write_report(csv_path: Path) -> Path:
    df = pd.read_csv(csv_path, keep_default_na=True)
    kind = "check" if "claim" in df else "ask"
    rows = df.to_dict("records")
    if kind == "ask":
        rows.sort(key=lambda r: (_score_ask(r), not _problem_ask(r)))
        cards = "".join(card_ask(r) for r in rows)
        cols = [c for c in ["correctness", "faithfulness", "correct_refusal", "tool_ok",
                            "correct_chunk", "general_knowledge", "web_used"] if c in df]
        table = df.groupby("category")[cols].mean().round(2)
        table.insert(0, "n", df.groupby("category").size())
        cats, n_bad = sorted(df["category"].unique()), sum(_problem_ask(r) for r in rows)
    else:
        rows.sort(key=lambda r: r.get("label") == r.get("pred"))
        cards = "".join(card_check(r) for r in rows)
        table = pd.crosstab(df["label"], df["pred"])
        cats, n_bad = sorted(df["label"].unique()), int((df["label"] != df["pred"]).sum())
    opts = "".join(f'<option value="{E(c)}">{E(c)}</option>' for c in cats)
    page = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Eval raporu</title>
<style>{CSS}</style></head><body>
<h1>Eval raporu: {kind}</h1><div class="mut">{E(csv_path.name)} · {len(df)} satır · {n_bad} sorunlu</div>
{table.to_html(classes="t", border=0, na_rep="")}
<div class="bar"><select id="cat"><option value="">tüm kategoriler</option>{opts}</select>
<label><input type="checkbox" id="bad"> sadece sorunlular</label>
<input id="q" placeholder="ara (ör. ticket, Velociraptor)"></div>
{cards}<script>{JS}</script></body></html>"""
    out = csv_path.with_suffix(".html")
    out.write_text(page, encoding="utf-8")
    return out


def print_top(csv_path: Path, n: int):
    df = pd.read_csv(csv_path)
    rows = df.to_dict("records")
    if "claim" in df:
        rows = [r for r in rows if r.get("label") != r.get("pred")][:n]
        for r in rows:
            print(f"\n{r['id']}  [{r['label']} → {r['pred']}]  {r['claim']}")
            for k in ["explanation", "evidence", "retrieved"]:
                if k in r and r[k] == r[k]:
                    print(f"  {k}: {r[k]}")
        return
    rows = sorted([r for r in rows if _problem_ask(r)], key=_score_ask)[:n]
    for r in rows:
        print("\n" + "=" * 90)
        print(f"{r['id']} [{r['category']}] {r['question']}")
        ans_col = "grounded" if "grounded" in r else "summary"      # eski CSV'lerde sadece özet var
        for k, label in [("expected", "BEKLENEN"), ("must", "OLMAZSA OLMAZ"), (ans_col, "CEVAP"),
                         ("general_text", "GENEL BİLGİ"), ("not_in_doc", "BELGEDE YOK"),
                         ("trace", "ARAÇLAR"), ("tickets", "KAYIT"), ("c_reason", "YARGIÇ"),
                         ("unsupported", "DAYANAKSIZ")]:
            v = r.get(k)
            if v == v and v not in (None, ""):
                print(f"  {label:14s} " + str(v).replace("\n", "\n" + " " * 17))
        print("  PUAN          " + "  ".join(f"{k}={r[k]}" for k in
                                             ["correctness", "faithfulness", "correct_refusal",
                                              "tool_ok", "ticket_opened"] if k in r and r[k] == r[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="varsayılan: en son *_ask.csv")
    ap.add_argument("--top", type=int, default=0, help="en kötü N satırı terminale bas")
    a = ap.parse_args()
    path = Path(a.csv) if a.csv else max(REPORTS.glob("*_ask.csv"), key=lambda p: p.stat().st_mtime)
    out = write_report(path)
    print(f"Rapor: {out}")
    if a.top:
        print_top(path, a.top)


if __name__ == "__main__":
    main()
