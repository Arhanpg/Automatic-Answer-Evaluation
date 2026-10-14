"""
backend/ollama_client.py -- minimal client for a local Ollama server.

No fake replies any more: if Ollama is not reachable you get an OllamaError
that says exactly what to do.
"""
import json
import time
import urllib.error
import urllib.request

from config import OLLAMA_EMBED_MODEL, OLLAMA_HOST, OLLAMA_MODEL


class OllamaError(RuntimeError):
    pass


_cache = (0.0, False)


def ollama_available(max_age=5.0):
    global _cache
    if time.time() - _cache[0] > max_age:
        try:
            urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=1.5)
            _cache = (time.time(), True)
        except Exception:
            _cache = (time.time(), False)
    return _cache[1]


def list_models():
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2) as r:
            return [m["name"] for m in json.loads(r.read().decode())["models"]]
    except Exception:
        return []


def _post(path, payload, timeout):
    req = urllib.request.Request(f"{OLLAMA_HOST}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")[:300]
        raise OllamaError(f"Ollama returned HTTP {e.code}: {body}  (is the model pulled? "
                          f"run: ollama pull {payload.get('model')})") from e
    except Exception as e:
        raise OllamaError(f"Cannot reach Ollama at {OLLAMA_HOST} ({e}). Install it from "
                          f"https://ollama.com/download, run `ollama serve`, then "
                          f"`ollama pull {OLLAMA_MODEL}`.") from e


def chat(messages, model=None, temperature=0.0, json_mode=False, timeout=300):
    payload = {"model": model or OLLAMA_MODEL, "messages": messages, "stream": False,
               "options": {"temperature": temperature}}
    if json_mode:
        payload["format"] = "json"
    return _post("/api/chat", payload, timeout).get("message", {}).get("content", "")


def embed(texts, model=None):
    model = model or OLLAMA_EMBED_MODEL
    try:
        out = _post("/api/embed", {"model": model, "input": list(texts)}, 120)
        if out.get("embeddings"):
            return out["embeddings"]
    except OllamaError:
        pass
    return [_post("/api/embeddings", {"model": model, "prompt": t}, 120)["embedding"] for t in texts]
