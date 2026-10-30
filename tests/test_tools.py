"""Araçların testleri. LLM YOK: saniyeler içinde biter, para harcamaz.

  python -m pytest -q

Deterministik kodun (araçlar, alıntı doğrulama) testi klasik yazılım testidir.
LLM'in davranışının testi ise eval'dir (python -m dino.eval). İkisi farklı şeyler ölçer.
"""
import json

import pytest

from dino.factcheck import verify_quote
from dino.tools import (CompareTimeline, FindSpecies, OpenGapTicket, ToolContext,
                        compare_timeline, execute, find_species, lookup_species, open_gap_ticket,
                        read_tickets, tool_specs)


def test_lookup_aliases():
    for name in ["T. rex", "trex", "Tyrannosaurus", "tyrannosaurus rex"]:
        assert lookup_species(name)["name"] == "Tyrannosaurus rex"
    assert lookup_species("brachio")["name"] == "Brachiosaurus"
    assert lookup_species("Godzilla") is None


def test_timeline_no_overlap():
    r = compare_timeline(CompareTimeline(species_a="Stegosaurus", species_b="T. rex"), ToolContext())
    assert r["lived_at_same_time"] is False
    assert r["gap_million_years"] == 82          # 150 Ma - 68 Ma


def test_timeline_overlap_and_order_does_not_matter():
    a = compare_timeline(CompareTimeline(species_a="Triceratops", species_b="T. rex"), ToolContext())
    b = compare_timeline(CompareTimeline(species_a="T. rex", species_b="Triceratops"), ToolContext())
    assert a["lived_at_same_time"] and b["lived_at_same_time"]
    assert a["gap_million_years"] == b["gap_million_years"] == 0


def test_timeline_unknown_species_gives_helpful_error():
    r = compare_timeline(CompareTimeline(species_a="Godzilla", species_b="T. rex"), ToolContext())
    assert "Unknown species" in r["error"] and "Stegosaurus" in r["error"]


def test_find_species_sort_and_filter():
    r = find_species(FindSpecies(diet="Carnivore", sort_by="mass_t", limit=1), ToolContext())
    assert r["rows"][0]["name"] == "Tyrannosaurus rex"
    r = find_species(FindSpecies(region="Mongolia"), ToolContext())
    assert [x["name"] for x in r["rows"]] == ["Velociraptor"]
    r = find_species(FindSpecies(sort_by="mass_t", limit=1), ToolContext())
    assert r["rows"][0]["name"] == "Argentinosaurus"


def test_execute_rejects_bad_arguments():
    out = json.loads(execute("find_species", json.dumps({"limit": 50}), ToolContext()))
    assert "limit must be between 1 and 10" in out["error"]
    out = json.loads(execute("find_species", json.dumps({"diet": "Vegan"}), ToolContext()))
    assert "Invalid arguments" in out["error"]
    out = json.loads(execute("drop_table", "{}", ToolContext()))
    assert "Unknown tool" in out["error"]
    out = json.loads(execute("find_species", "{bozuk json", ToolContext()))
    assert "Invalid arguments" in out["error"]


def test_ticket_dedup_and_limit(tmp_path):
    path = tmp_path / "tickets.jsonl"
    args = OpenGapTicket(question="What color was Triceratops?",
                         missing_topic="Triceratops skin color", priority="low")
    r1 = open_gap_ticket(args, ToolContext(tickets_path=path))
    assert r1["status"] == "created"
    # aynı konu, kelime sırası farklı → yeni kayıt açılmaz
    same = args.model_copy(update={"missing_topic": "skin color of Triceratops"})
    r2 = open_gap_ticket(same, ToolContext(tickets_path=path))
    assert r2 == {"status": "already_open", "ticket_id": r1["ticket_id"]}
    # bir istekte en fazla 1 kayıt
    ctx = ToolContext(tickets_path=path)
    open_gap_ticket(args.model_copy(update={"missing_topic": "T. rex teeth count"}), ctx)
    r4 = open_gap_ticket(args.model_copy(update={"missing_topic": "Ankylosaurus lifespan"}), ctx)
    assert "already opened" in r4["error"]
    assert len(read_tickets(path)) == 2


def test_tool_specs_are_strict():
    specs = {s["function"]["name"]: s["function"] for s in tool_specs()}
    assert set(specs) == {"search_docs", "find_species", "compare_timeline", "compare_size", "open_gap_ticket",
                          "export_document", "species_table_file"}
    assert all(f["strict"] for f in specs.values())
    assert specs["find_species"]["parameters"]["properties"]["diet"]["anyOf"][0]["enum"]


def test_verify_quote():
    src = "Older popular estimates gave T. rex running speeds of 60-70 km/h. Biomechanical studies"
    assert verify_quote("gave T. rex running speeds of 60-70 km/h", src)
    assert verify_quote("Older popular estimates gave T rex running speeds of 60-70 km/h", src)
    assert not verify_quote("T. rex could run at 70 km/h easily", src)


def test_normalize_source():
    from dino.retrieval import normalize_source
    assert normalize_source("4.2") == "guide:s4.2"
    assert normalize_source("[s4.10]") == "guide:s4.10"
    assert normalize_source("tyrannosaurus:72") == "tyrannosaurus:72"


def test_wiki_clean_and_split():
    from dino.parsers import clean, split_text
    assert clean("Dinosaurs are reptiles[note 1] of the clade[12][a] ( tyrannos , x)") == \
        "Dinosaurs are reptiles of the clade (tyrannos, x)"
    parts = split_text("A" * 900 + ".\n" + "B" * 900 + ".\nkısa.", max_chars=1200)
    assert len(parts) == 2 and parts[-1].endswith("kısa.")       # kısa kuyruk öncekine eklenir


def test_compare_size_keeps_ranges():
    from dino.tools import CompareSize, compare_size
    r = compare_size(CompareSize(taxon_a="Argentinosaurus", taxon_b="T. rex", measure="mass"), ToolContext())
    assert r["bigger"] == "Argentinosaurus" and r["times"] == "8.1-9.4"      # 65/8 – 75/8
    r = compare_size(CompareSize(taxon_a="Velociraptor", taxon_b="T. rex", measure="length"), ToolContext())
    assert r["bigger"] == "Tyrannosaurus rex" and r["times"] == "6.0"        # sıra ters verilse de
    r = compare_size(CompareSize(taxon_a="Godzilla", taxon_b="T. rex", measure="mass"), ToolContext())
    assert "Unknown taxon" in r["error"]
    r = compare_size(CompareSize(taxon_a="Triceratops", taxon_b="T. rex", measure="mass"), ToolContext())
    assert r["comparable"] and r["bigger"] is None                         # 6-12 t vs 8 t: örtüşüyor


def test_search_filters_and_dynamic_schema(monkeypatch):
    import numpy as np
    from dino import retrieval
    from dino.tools import schema_for
    chunks = [{"id": "guide:s1", "doc": "Guide", "source": "guide", "text": "a"},
              {"id": "tyrannosaurus:1", "doc": "Tyrannosaurus", "source": "wikipedia", "text": "b"},
              {"id": "sauropoda:1", "doc": "Sauropoda", "source": "wikipedia", "text": "c"}]
    E = np.eye(3, dtype=np.float32)
    monkeypatch.setattr(retrieval, "encode", lambda t, kind: np.array([[0.1, 0.9, 0.5]]))
    monkeypatch.setattr(retrieval.Retriever, "queries_for", lambda self, q, mode="always": [q])
    r = retrieval.Retriever(chunks, E)
    assert r.search("x", 3)[0] == "tyrannosaurus:1"
    assert r.search("x", 3, source="guide") == ["guide:s1"]
    assert r.search("x", 3, doc="Sauropoda") == ["sauropoda:1"]
    assert r.search("x", 3, source="guide", doc="Sauropoda") == []
    M = schema_for("search_docs", ToolContext(retriever=r))
    assert M.model_validate({"query": "q", "source": None, "article": "Sauropoda"})
    with pytest.raises(Exception):
        M.model_validate({"query": "q", "source": None, "article": "Godzilla"})


def test_extract_validation():
    from dino.extract import SizeFact, validate
    text = "The heaviest known sauropods include Argentinosaurus (65 to 80 metric tons)."
    ok, _ = validate(SizeFact(taxon="Argentinosaurus", quote="Argentinosaurus (65 to 80 metric tons)",
                              mass_min=65, mass_max=80, mass_unit="t"), text)
    assert ok["mass_t_min"] == 65 and ok["mass_t_max"] == 80
    bad, issues = validate(SizeFact(taxon="Argentinosaurus", quote="Argentinosaurus (65 to 80 metric tons)",
                                    mass_min=90, mass_max=100, mass_unit="t"), text)
    assert bad is None and "sayı alıntıda yok" in issues[0]
    fake, issues = validate(SizeFact(taxon="X", quote="X weighed 500 tonnes", mass_min=500,
                                     mass_unit="t"), text)
    assert fake is None and issues == ["alıntı chunk'ta yok"]
    kg, _ = validate(SizeFact(taxon="Velociraptor", quote="weighing about 15 kg", mass_min=15,
                              mass_unit="kg"), "Velociraptor ... weighing about 15 kg (33 lb)")
    assert kg["mass_t_min"] == 0.015                         # birim dönüşümü kodda


def test_guide_headings_must_be_sequential():
    from dino.ingest import sequential_heads
    text = "1 Intro\nx\n2 Timeline\ny\n12 Carnivore North America\n2.1 Triassic Period\nz\n3 Classification\n"
    assert [h[0] for h in sequential_heads(text)] == ["1", "2", "2.1", "3"]


def test_eval_metrics_accept_valid_alternatives():
    from dino.eval import is_gold, tools_ok
    g = {"gold": ["4"], "gold_docs": []}
    assert is_gold("guide:s4.1", g, None) and is_gold("guide:s4", g, None)
    assert not is_gold("guide:s5.1", g, None) and not is_gold("guide:s40", g, None)
    assert tools_ok([["find_species", "compare_timeline"]], ["compare_timeline"]) == 1.0
    assert tools_ok(["compare_size"], ["find_species"]) == 0.0


def test_extract_aggregate_prefers_own_article_and_median():
    from dino.extract import aggregate
    rows = [{"name": "Velociraptor", "chunk": "velociraptor:10", "mass_t_min": 0.0141, "mass_t_max": 0.0197,
             "length_m_min": None, "length_m_max": None, "start_ma": None, "end_ma": None},
            {"name": "Velociraptor", "chunk": "dromaeosauridae:3", "mass_t_min": 0.09, "mass_t_max": 0.09,
             "length_m_min": None, "length_m_max": None, "start_ma": None, "end_ma": None},
            {"name": "Spinosaurus", "chunk": "theropoda:13", "mass_t_min": 7, "mass_t_max": 7,
             "length_m_min": None, "length_m_max": None, "start_ma": None, "end_ma": None},
            {"name": "Spinosaurus", "chunk": "spinosauridae:4", "mass_t_min": 6, "mass_t_max": 7.4,
             "length_m_min": None, "length_m_max": None, "start_ma": None, "end_ma": None},
            {"name": "Spinosaurus", "chunk": "dinosaur:52", "mass_t_min": 7, "mass_t_max": 20.9,
             "length_m_min": None, "length_m_max": None, "start_ma": None, "end_ma": None}]
    df = aggregate(rows).set_index("name")
    assert df.loc["Velociraptor", "mass_t_max"] == 0.0197        # kendi makalesi kazanır, 90 kg girmez
    assert df.loc["Spinosaurus", "mass_t_max"] == 7.4            # medyan: 20.9 uç değeri satırı bozmaz


def test_live_lookup_and_reference_compare(monkeypatch, tmp_path):
    """Canlı Wikipedia yolu: ağ ve LLM taklit edilir, doğrulama ve karşılaştırma gerçek kodla çalışır."""
    from types import SimpleNamespace
    from dino import llm, tools, wiki_live
    from dino.config import get_settings
    from dino.extract import Extraction, SizeFact
    from dino.tools import CompareSize, CompareTimeline, compare_size, compare_timeline

    s = get_settings()
    monkeypatch.setattr(s, "live_tools", True)
    monkeypatch.setattr(wiki_live, "_cache_path", lambda: tmp_path / "live_cache.jsonl")
    page = ("Ceratosaurus is a genus of theropod dinosaur that lived in the Late Jurassic.\n"
            "Ceratosaurus nasicornis was about 5.3 metres long and weighed between 418 and 980 kg.")
    calls = {"http": 0}

    def fake_get(params):
        calls["http"] += 1
        t = params["titles"]
        if t.lower() != "ceratosaurus":
            return {"query": {"pages": {"-1": {"missing": ""}}}}
        return {"query": {"pages": {"1": {"title": "Ceratosaurus", "extract": page,
                                          "fullurl": "https://en.wikipedia.org/wiki/Ceratosaurus"}}}}

    def fake_parse(messages, schema, **kw):
        text = messages[-1]["content"]
        facts = []
        if "5.3 metres" in text:
            facts = [SizeFact(taxon="Ceratosaurus", quote="was about 5.3 metres long and weighed between 418 and 980 kg",
                              length_min=5.3, length_max=5.3, length_unit="m",
                              mass_min=418, mass_max=980, mass_unit="kg")]
        return SimpleNamespace(parsed=Extraction(facts=facts))

    monkeypatch.setattr(wiki_live, "_http_get", fake_get)
    monkeypatch.setattr(llm, "parse", fake_parse)
    tools._LIVE_ERRORS.clear()

    # önce belge: arama yapılmadan canlıya gidilmez, model yönlendirilir
    r = compare_size(CompareSize(taxon_a="Ceratosaurus", taxon_b="T. rex", measure="length"), ToolContext())
    assert "search the documents" in r["error"] and calls["http"] == 0
    r = compare_size(CompareSize(taxon_a="Ceratosaurus", taxon_b="T. rex", measure="length"), ToolContext(searched=True))
    assert r["bigger"] == "Tyrannosaurus rex" and r["links"] == {"web:Ceratosaurus": "https://en.wikipedia.org/wiki/Ceratosaurus"}
    assert "wikipedia-live" in r["origins"]
    # ikinci kez: önbellekten, ağa gidilmez
    compare_size(CompareSize(taxon_a="Ceratosaurus", taxon_b="T. rex", measure="mass"), ToolContext(searched=True))
    assert calls["http"] == 1
    # referans nesne: kaç fil?
    r = compare_size(CompareSize(taxon_a="Argentinosaurus", taxon_b="elephant", measure="mass"), ToolContext(searched=True))
    assert r["bigger"] == "Argentinosaurus" and "ref:african_elephant" in r["links"]
    # insanlarla zaman farkı
    r = compare_timeline(CompareTimeline(species_a="T. rex", species_b="insan"), ToolContext())
    assert r["lived_at_same_time"] is False and r["gap_million_years"] == 65.7
    # olmayan sayfa: okunabilir hata
    r = compare_size(CompareSize(taxon_a="Godzillasaurus", taxon_b="T. rex", measure="mass"), ToolContext(searched=True))
    assert "No Wikipedia article" in r["error"]


def test_live_tools_off_by_default():
    from dino.tools import CompareSize, compare_size
    r = compare_size(CompareSize(taxon_a="Ceratosaurus", taxon_b="T. rex", measure="mass"), ToolContext())
    assert "Unknown taxon" in r["error"]


def test_find_species_falls_back_to_live(monkeypatch):
    from dino import tools, wiki_live
    monkeypatch.setattr(tools.get_settings(), "live_tools", True)
    monkeypatch.setattr(wiki_live, "lookup_taxon", lambda n: {
        "name": "Ceratosaurus", "start_ma": 153.0, "end_ma": 148.0, "length_m_min": 5.3,
        "length_m_max": 5.7, "mass_t_min": 0.4, "mass_t_max": 0.7, "section": "web:Ceratosaurus",
        "origin": "wikipedia-live", "url": "https://en.wikipedia.org/wiki/Ceratosaurus"})
    tools._LIVE_ERRORS.clear()
    r = find_species(FindSpecies(name="Ceratosaurus"), ToolContext())
    assert not r["rows"] and "search the documents" in r["note"]          # önce belge
    r = find_species(FindSpecies(name="Ceratosaurus"), ToolContext(searched=True))
    assert r["rows"][0]["origin"] == "wikipedia-live"
    assert r["links"] == {"web:Ceratosaurus": "https://en.wikipedia.org/wiki/Ceratosaurus"}


def test_repeated_call_is_blocked():
    ctx = ToolContext()
    args = json.dumps({"diet": "Carnivore", "limit": 1})
    assert "rows" in json.loads(execute("find_species", args, ctx))
    assert "already made this exact call" in json.loads(execute("find_species", args, ctx))["error"]


def test_web_only_document_claim_moves_to_web_layer():
    from dino.agent import _check_sources
    from dino.schemas import Claim, LayeredAnswerV2
    a = LayeredAnswerV2.model_validate({
        "summary": "x", "corrections": [], "inference": [], "web_facts": [], "debated": [],
        "not_in_document": [], "general_knowledge": [], "coverage": "full", "on_topic": True,
        "document": [{"text": "Allosaurus 8.5 m", "sources": ["web:Allosaurus"]},
                     {"text": "T. rex 12 m", "sources": ["guide:s4.1"]}]})
    ctx = ToolContext(seen_chunks=["guide:s4.1"], tool_outputs=['[find_species] {"source": "web:Allosaurus"}'])
    _check_sources(a, ctx, [])
    assert [c.text for c in a.document] == ["T. rex 12 m"]
    assert [c.text for c in a.web_facts] == ["Allosaurus 8.5 m"]
    assert a.coverage == "partial"                     # web gerektiyse belge kapsamı 'full' olamaz


def test_gated_call_can_be_repeated_after_search(monkeypatch):
    from dino import tools, wiki_live
    monkeypatch.setattr(tools.get_settings(), "live_tools", True)
    monkeypatch.setattr(wiki_live, "lookup_taxon", lambda n: {
        "name": "Ceratosaurus", "start_ma": 153.0, "end_ma": 148.0, "length_m_min": 6.0,
        "length_m_max": 6.7, "mass_t_min": 0.45, "mass_t_max": 0.67, "section": "web:Ceratosaurus",
        "origin": "wikipedia-live", "url": "https://en.wikipedia.org/wiki/Ceratosaurus"})
    ctx, args = ToolContext(), json.dumps({"name": "Ceratosaurus"})
    assert "search the documents" in json.loads(execute("find_species", args, ctx))["note"]
    ctx.searched = True                                    # model belgeleri aradı
    assert json.loads(execute("find_species", args, ctx))["rows"][0]["origin"] == "wikipedia-live"


def test_web_number_in_document_layer_is_caught():
    from types import SimpleNamespace
    from dino.agent import _misattributed
    from dino.schemas import LayeredAnswerV2
    base = {"summary": "x", "corrections": [], "inference": [], "debated": [], "not_in_document": [],
            "general_knowledge": [], "coverage": "partial", "on_topic": True, "document": [],
            "web_facts": [{"text": "Ceratosaurus was 6-6.7 m long.", "sources": ["web:Ceratosaurus"]}]}
    ret = SimpleNamespace(by_id={"ceratosauria:3": {"text": "Ceratosaurus was moderately large."},
                                 "allosauridae:2": {"text": "Allosaurus was about 8 meters long."}})
    ctx = ToolContext(retriever=ret, seen_chunks=["ceratosauria:3", "allosauridae:2"])
    bad = LayeredAnswerV2.model_validate({**base, "document_says": "The documents say Ceratosaurus was 6-6.7 m long."})
    assert _misattributed(bad, ctx) == ["6", "6.7"]
    ok = LayeredAnswerV2.model_validate({**base, "document_says": "The documents only call Ceratosaurus "
                                                                   "moderately large; Allosaurus about 8 m."})
    assert _misattributed(ok, ctx) == []


def test_follow_up_is_condensed_and_history_is_in_context(monkeypatch):
    from types import SimpleNamespace
    from dino import agent
    from dino.schemas import LayeredAnswer, Standalone
    seen = []

    def fake_parse(messages, schema, tag="", **kw):
        seen.append((tag, messages))
        if schema is Standalone:
            return SimpleNamespace(parsed=Standalone(standalone_question="Did Ceratosaurus eat Allosaurus?"))
        ans = LayeredAnswer(corrections=[], document=[], inference=[], not_in_document=[],
                            general_knowledge=[], summary="Belgelerde bu yok.")
        return SimpleNamespace(parsed=ans, tool_calls=None)
    monkeypatch.setattr(agent.llm, "parse", fake_parse)

    hist = [{"question": "Was Ceratosaurus bigger than Allosaurus?", "answer": "No, Allosaurus was bigger."}]
    r = agent.ask("Peki onu yemiş miydi?", retriever=None, history=hist)
    assert r["standalone_question"] == "Did Ceratosaurus eat Allosaurus?"
    tag, msgs = seen[1]                                     # 1. çağrı condense, 2. çağrı agent
    assert msgs[-1]["content"].startswith("Peki onu yemiş miydi?")          # dil için orijinal mesaj
    assert "Did Ceratosaurus eat Allosaurus?" in msgs[-1]["content"]       # arama için net soru
    assert {"role": "assistant", "content": "No, Allosaurus was bigger."} in msgs

    seen.clear()                                            # geçmiş yoksa ek çağrı yok
    r = agent.ask("Did T. rex have lips?", retriever=None)
    assert r["standalone_question"] is None and seen[0][0] == "agent"


def test_agent_queries_skip_llm_rewrite(monkeypatch):
    """Agent zaten İngilizce sorgu yazar: aramada ikinci kez LLM ile yeniden yazmak ~1-1.5 sn kaybettirir."""
    import numpy as np
    from dino import retrieval
    calls = []
    monkeypatch.setattr(retrieval, "rewrite", lambda q: calls.append(q) or ("rewritten query",))
    r = retrieval.Retriever([{"id": "a:1", "doc": "A", "source": "wikipedia", "text": "x"}], np.eye(1, dtype=np.float32))
    assert r.queries_for("Ceratosaurus body length", "auto") == ["Ceratosaurus body length"] and calls == []
    assert r.queries_for("Ceratosaurus boyu ne kadardı?", "auto") == ["rewritten query"]        # Türkçe kaçtıysa
    assert r.queries_for("How long was Ceratosaurus?") == ["How long was Ceratosaurus?", "rewritten query"]  # ham soru
