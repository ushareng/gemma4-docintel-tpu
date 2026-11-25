"""Minimal stand-in for the vLLM OpenAI server, for testing the app and benchmark
without a TPU. Simulates prefill cost ∝ uncached prompt length, a prefix cache,
per-token decode latency, structured outputs (schema-valid dummy JSON) and /metrics.

  uvicorn tests.fake_vllm:app --port 8000
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

app = FastAPI()
CACHE: set[str] = set()
HITS = QUERIES = 0
PREFILL_S_PER_TOK = 2e-5
DECODE_S_PER_TOK = 1e-3
BLOCK = 256  # chars per simulated cache block


def _prompt_text(body: dict) -> str:
    parts = []
    for m in body.get("messages", []):
        c = m["content"]
        if isinstance(c, str):
            parts.append(c)
        else:
            parts += [p.get("text", "<img>" * 200) for p in c]
    return "\n".join(parts)


def _prefill(text: str) -> tuple[int, float]:
    global HITS, QUERIES
    n_tok = max(1, len(text) // 4)
    blocks = [text[: i + BLOCK] for i in range(0, len(text), BLOCK)]
    hit = 0
    for b in blocks:
        h = hashlib.md5(b.encode()).hexdigest()
        if h in CACHE:
            hit += 1
        else:
            break
    if app.state.prefix_caching:
        for b in blocks:
            CACHE.add(hashlib.md5(b.encode()).hexdigest())
    hit_tok = n_tok * hit // max(1, len(blocks))
    HITS += hit_tok
    QUERIES += n_tok
    return n_tok, (n_tok - hit_tok) * PREFILL_S_PER_TOK + 0.01


def _dummy(schema: dict, defs: dict) -> object:
    if "$ref" in schema:
        return _dummy(defs[schema["$ref"].split("/")[-1]], defs)
    if "anyOf" in schema:
        return _dummy(schema["anyOf"][0], defs)
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    if t == "object":
        return {k: _dummy(v, defs) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [_dummy(schema["items"], defs)]
    return {"string": "sample", "number": 1.0, "integer": 1, "boolean": True}.get(t)


import os
app.state.prefix_caching = os.environ.get("FAKE_PREFIX_CACHING", "on") == "on"


@app.get("/health")
async def health():
    return PlainTextResponse("ok")


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "gemma4-e4b", "object": "model"}]}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(f'vllm:prefix_cache_hits_total{{model_name="gemma4-e4b"}} {HITS}\n'
                             f'vllm:prefix_cache_queries_total{{model_name="gemma4-e4b"}} {QUERIES}\n')


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    n_prompt, prefill_s = _prefill(_prompt_text(body))
    rf = body.get("response_format")
    if rf and rf.get("type") == "json_schema":
        schema = rf["json_schema"]["schema"]
        text = json.dumps(_dummy(schema, schema.get("$defs", {})))
    else:
        text = "This is a simulated answer from the fake server. " * 40
    n_out = body.get("max_tokens", 16)
    words = text.split(" ")
    if not body.get("ignore_eos"):
        n_out = min(n_out, len(words))

    if not body.get("stream"):
        await asyncio.sleep(prefill_s + n_out * DECODE_S_PER_TOK)
        return JSONResponse({
            "id": "x", "object": "chat.completion", "created": int(time.time()), "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out,
                      "total_tokens": n_prompt + n_out}})

    async def gen():
        await asyncio.sleep(prefill_s)
        for i in range(n_out):
            if i:
                await asyncio.sleep(DECODE_S_PER_TOK)
            tok = words[i % len(words)] + " "
            yield "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": tok}}]}) + "\n\n"
        yield "data: " + json.dumps({"choices": [], "usage": {
            "prompt_tokens": n_prompt, "completion_tokens": n_out,
            "total_tokens": n_prompt + n_out}}) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
