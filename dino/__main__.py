"""Tek giriş noktası: komut ezberlemek yok.

    python -m dino

Menüden seç. Deney ayarlarını (.env'i elle düzenlemeden) menü kendisi verir, ölçüm bitince
rapor tarayıcıda kendiliğinden açılır.
"""
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# Deney adımları: her biri bir öncekinin üzerine TEK bir değişiklik ekler (ablation).
STEPS = {
    "1": ("Ölçüm v2 (taban çizgisi)", {}, False),
    "2": ("+ Tablo düzeltmesi (extract v2)", {}, True),
    "3": ("+ Arama düzeltmesi (spillover)", {"SEARCH_SPILLOVER": "true"}, False),
    "4": ("+ Şema v2 (coverage, kaydı kod açar)", {"SEARCH_SPILLOVER": "true", "ANSWER_SCHEMA_V2": "true"}, False),
    "5": ("+ Canlı Wikipedia ve referans karşılaştırmaları",
          {"SEARCH_SPILLOVER": "true", "ANSWER_SCHEMA_V2": "true", "LIVE_TOOLS": "true"}, False),
}
ALL_ON = STEPS["5"][1]


def run(args, env_extra=None) -> int:
    env = {**os.environ, **(env_extra or {})}       # ortam değişkeni .env'den önceliklidir
    print(f"\n> python {' '.join(args)}" + (f"   [{', '.join(f'{k}={v}' for k, v in env_extra.items())}]"
                                            if env_extra else "") + "\n")
    return subprocess.call([PY, *args], cwd=ROOT, env=env)


def open_latest_report(kind="ask"):
    files = sorted((ROOT / "reports").glob(f"*_{kind}.html"), key=lambda p: p.stat().st_mtime)
    if files:
        webbrowser.open(files[-1].as_uri())
        print(f"Rapor açıldı: {files[-1].name}")
    else:
        print("Henüz rapor yok. Önce bir ölçüm yap.")


def experiment():
    print("\nHangi deney adımı? (her adım öncekilerin üzerine bir değişiklik ekler)")
    for k, (name, _, _) in STEPS.items():
        print(f"  {k}) {name}")
    k = input("Seçim: ").strip()
    if k not in STEPS:
        return
    name, flags, needs_extract = STEPS[k]
    scope = input("Sadece sorular mı (s), sorular + iddialar mı (t)? [s]: ").strip().lower() or "s"
    limit = input("Kaç soru? (boş = hepsi, deneme için 5): ").strip()
    if needs_extract and run(["-m", "dino.extract"]) != 0:
        return
    cmd = ["-m", "dino.eval", "all" if scope == "t" else "ask", "--note", f"adım {k}: {name}"]
    if limit:
        cmd += ["--limit", limit]
    if run(cmd, flags) != 0:
        return
    if k == "5":
        # Yeni yetenek için ayrı soru seti: aynı 5 soru önce canlı araçlar KAPALI, sonra AÇIK.
        off = {**flags, "LIVE_TOOLS": "false"}
        run(["-m", "dino.eval", "live", "--note", "canlı araçlar kapalı"], off)
        run(["-m", "dino.eval", "live", "--note", "canlı araçlar açık"], flags)
        open_latest_report("live")
    else:
        open_latest_report("ask")


APP_FLAGS = dict(ALL_ON)     # uygulama en güncel sistemle açılır (bütün düzeltmeler + canlı araçlar)


def app():
    print("\nUygulama başlıyor (bütün düzeltmeler + canlı Wikipedia açık). Kapatmak için Ctrl+C.")
    p = subprocess.Popen([PY, "-m", "uvicorn", "dino.api:app", "--port", "8000"], cwd=ROOT,
                         env={**os.environ, **APP_FLAGS})
    for _ in range(90):                               # model ve indeks yüklenene kadar bekle
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1)
            webbrowser.open("http://127.0.0.1:8000/")
            print("\nTarayıcıda açıldı: http://127.0.0.1:8000/")
            break
        except Exception:
            time.sleep(1)
    try:
        p.wait()
    except KeyboardInterrupt:
        p.terminate()
        print("\nUygulama kapatıldı.")


def wiki_test():
    sys.path.insert(0, str(ROOT))
    from dino.wiki_live import fetch_page
    try:
        p = fetch_page("Ceratosaurus")
        print(f"\nBağlantı tamam: {p['url']}\n{p['text'][:200]}…")
    except Exception as e:
        print(f"\nWikipedia'ya ulaşılamadı: {type(e).__name__}: {str(e)[:200]}")
        if "SSL" in str(e) or "CERTIFICATE" in str(e).upper():
            print("Ofis SSL denetimi. Çözüm: python -m pip install truststore  (sonra tekrar dene)")


def live_only():
    note = input("Not (boş bırakabilirsin): ").strip() or "canlı set"
    if run(["-m", "dino.eval", "live", "--note", note], ALL_ON) == 0:
        open_latest_report("live")


MENU = """
================ Dino Bilgi Asistanı ================
  1) Uygulamayı aç (sohbet sayfası)
  2) Deney / ölçüm çalıştır
  3) Son raporu aç
  4) Testleri çalıştır (LLM yok, ücretsiz)
  5) İndeksi yeniden kur (sadece doküman değişince)
  6) Wikipedia bağlantısını test et (canlı araçlar için)
  7) Sadece canlı soru setini ölç (5 soru, ~2 dk)
  0) Çıkış
====================================================="""


def main():
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")   # ofis proxy'si yerel adresi yakalamasın
    actions = {"1": app, "2": experiment, "3": open_latest_report,
               "4": lambda: run(["-m", "pytest", "-q"]), "5": lambda: run(["-m", "dino.ingest"]),
               "6": wiki_test, "7": live_only}
    while True:
        print(MENU)
        k = input("Seçim: ").strip()
        if k == "0":
            break
        if k in actions:
            actions[k]()


if __name__ == "__main__":
    main()
