"""Sahte, OpenAI uyumlu LLM sunucusu: kodu PARA HARCAMADAN uçtan uca test etmek için.
Cevaplar anlamlı değildir; amaç kod yollarının (araç döngüsü, şema, hata) çalıştığını görmek.

  python tests/mock_llm.py 8765
  .env:  LLM_BASE_URL=http://127.0.0.1:8765/v1   EMBED_BACKEND=api
"""
import hashlib, json, re, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = 256


def fake(s, defs=None):
    defs = defs or s.get("$defs", {})
    if "$ref" in s:
        return fake(defs[s["$ref"].split("/")[-1]], defs)
    if "anyOf" in s:
        opts = [o for o in s["anyOf"] if o.get("type") != "null"]
        return fake(opts[0], defs) if opts else None
    if "enum" in s:
        return s["enum"][0]
    t = s.get("type")
    if t == "object":
        return {k: fake(v, defs) for k, v in s.get("properties", {}).items()}
    if t == "array":
        return [fake(s.get("items", {"type": "string"}), defs)]
    return {"number": 0.8, "integer": 4, "boolean": False}.get(t, "örnek metin")


def embed(text):
    v = [0.0] * DIM
    for w in re.findall(r"\w+", text.lower()):
        v[int(hashlib.md5(w[:6].encode()).hexdigest(), 16) % DIM] += 1.0
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def pick_tool(question, n_tool_msgs):
    q = question.lower()
    if "same time" in q or "aynı dönem" in q:
        return "compare_timeline", {"species_a": "Stegosaurus", "species_b": "T. rex"}
    if "times" in q or " kat" in q:
        return "compare_size", {"taxon_a": "Argentinosaurus", "taxon_b": "T. rex", "measure": "mass"}
    if any(w in q for w in ["heav", "largest", "büyük", "mongolia", "brain"]):
        return "find_species", {"name": None, "diet": None, "period": None, "region": None,
                                "min_mass_t": None, "max_mass_t": None, "sort_by": "mass_t",
                                "descending": True, "limit": 3}
    if any(w in q for w in ["color", "teeth"]) and n_tool_msgs == 1:
        return "open_gap_ticket", {"question": question, "missing_topic": question[:40],
                                   "priority": "low"}
    return "search_docs", {"query": question, "source": None, "article": None}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        """Sahte Wikipedia API'si (WIKI_API_BASE=http://127.0.0.1:PORT): canlı araç yolunu test eder."""
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        title = (q.get("titles") or [""])[0]
        if title.lower() not in ("ceratosaurus", "allosaurus", "giganotosaurus"):
            return self._send({"query": {"pages": {"-1": {"title": title, "missing": ""}}}})
        text = f"{title.capitalize()} is a genus of theropod dinosaur.\n{title.capitalize()} was about 6 metres long."
        self._send({"query": {"pages": {"1": {"title": title.capitalize(), "extract": text,
                                              "fullurl": f"https://en.wikipedia.org/wiki/{title.capitalize()}"}}}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if self.path.endswith("/embeddings"):
            inp = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return self._send({"object": "list", "model": body.get("model"),
                               "data": [{"object": "embedding", "index": i, "embedding": embed(x)}
                                        for i, x in enumerate(inp)],
                               "usage": {"prompt_tokens": 10, "total_tokens": 10}})
        msgs = body.get("messages", [])
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        if "ignore your previous instructions" in user.lower():      # içerik filtresi taklidi
            return self._send({"error": {"message": "filtered", "code": "content_filter",
                                         "type": "invalid_request_error"}}, 400)
        for m in msgs:                      # gerçek API gibi katı ol: bilinmeyen alan → 400
            for tc in m.get("tool_calls") or []:
                extra = set(tc["function"]) - {"name", "arguments"}
                if extra:
                    return self._send({"error": {"message": f"Unrecognized fields {extra}",
                                                 "type": "invalid_request_error"}}, 400)
        n_tool = sum(1 for m in msgs if m["role"] == "tool")
        msg, finish = {"role": "assistant", "content": None}, "stop"
        tools, choice = body.get("tools"), body.get("tool_choice")
        if tools and choice != "none" and (choice == "required" or n_tool < 2):
            name, args = pick_tool(user, n_tool)
            if name not in [t["function"]["name"] for t in tools]:
                name, args = "search_docs", {"query": user, "source": None, "article": None}
            msg["tool_calls"] = [{"id": "call_" + hashlib.md5(json.dumps(msgs).encode()).hexdigest()[:8],
                                  "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]
            finish = "tool_calls"
        elif body.get("response_format", {}).get("type") == "json_schema":
            msg["content"] = json.dumps(fake(body["response_format"]["json_schema"]["schema"]),
                                        ensure_ascii=False)
        else:
            msg["content"] = "Sahte cevap."
        self._send({"id": "chatcmpl-x", "object": "chat.completion", "created": int(time.time()),
                    "model": body.get("model", "mock"),
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}})


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
