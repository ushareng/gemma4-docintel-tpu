"""Streaming load generator for the vLLM OpenAI server.

Measures, per request: time-to-first-token (TTFT), time per output token (TPOT,
= (E2E - TTFT) / (output_tokens - 1)) and end-to-end latency; per run: request,
output-token and total-token throughput, plus vLLM's prefix-cache hit rate
scraped from /metrics.

Workloads
  prefill  – long SHARED document context (the same for every request) + a short
             question, 64 output tokens. Dominated by prefill; prefix caching can
             skip recomputing the shared document.
  unique   – same as prefill but each request gets a unique preamble before the
             document, so the prefix cache cannot help (control group).
  decode   – short prompt, long generation (512 tokens, EOS ignored). Dominated
             by decode; prefix caching should barely matter.

Example
  python -m bench.bench --workload prefill --concurrency 1 4 16 --tag pc_on
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent

QUESTIONS = [
    "When does the agreement start and how long is the initial term?",
    "What is the monthly retainer?",
    "What hourly rate applies to additional consulting?",
    "Within how many days must invoices be paid?",
    "What interest applies to overdue amounts?",
    "When may the provider suspend services?",
    "What availability does the provider guarantee?",
    "What service credit applies if availability is missed?",
    "What is the cap on service credits per quarter?",
    "In which regions is client data processed?",
    "How quickly is client data deleted after termination?",
    "Who owns fine-tuned model weights?",
    "What does the provider keep ownership of?",
    "What is each party's liability limited to?",
    "Which law governs the agreement?",
    "Which courts have jurisdiction?",
    "How much notice is needed to stop auto-renewal?",
    "When are invoices issued?",
    "Who are the parties to the agreement?",
    "Is GST included in the retainer?",
]
TOPICS = ["the history of tensor processing hardware", "how paged attention manages KV cache",
          "designing a document-understanding pipeline", "the economics of LLM inference",
          "trade-offs between prefill and decode", "structured generation with JSON schemas",
          "retrieval-augmented generation with BM25", "systolic arrays and matrix multiplication"]


# ------------------------------------------------------------------ prompts
def load_context(target_words: int) -> str:
    """Shared document text: the sample agreement, repeated to reach target length."""
    pdf = ROOT / "samples" / "service_agreement.pdf"
    if pdf.exists():
        import pymupdf
        with pymupdf.open(pdf) as d:
            base = "\n\n".join(f"[page {i + 1}]\n{p.get_text().strip()}" for i, p in enumerate(d))
    else:
        base = " ".join(["Lorem ipsum dolor sit amet, consectetur adipiscing elit."] * 200)
    words = base.split()
    reps = max(1, -(-target_words // len(words)))
    return " ".join((words * reps)[:target_words])


def build_requests(workload: str, n: int, context_words: int, output_tokens: int,
                   model: str, seed: int = 0) -> list[dict]:
    rnd = random.Random(seed)
    reqs = []
    if workload in ("prefill", "unique"):
        ctx = load_context(context_words)
        for i in range(n):
            preamble = f"Request id {uuid.uuid4()}.\n" if workload == "unique" else ""
            reqs.append({
                "model": model,
                "messages": [
                    {"role": "system", "content": preamble +
                     "Answer questions about the document below concisely.\n\n" + ctx},
                    {"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]},
                ],
                "max_tokens": output_tokens, "temperature": 0.0,
            })
    elif workload == "decode":
        for i in range(n):
            reqs.append({
                "model": model,
                "messages": [{"role": "user", "content":
                              f"Write a long, detailed technical essay about {rnd.choice(TOPICS)}. "
                              f"(variant {i})"}],
                "max_tokens": output_tokens, "temperature": 0.7,
                "ignore_eos": True, "min_tokens": output_tokens,  # vLLM extensions: fixed length
            })
    else:
        raise ValueError(workload)
    for r in reqs:
        r["stream"] = True
        r["stream_options"] = {"include_usage": True}
        r["chat_template_kwargs"] = {"enable_thinking": False}
    return reqs


# ------------------------------------------------------------------ measurement
@dataclass
class Result:
    ok: bool
    ttft: float = 0.0
    e2e: float = 0.0
    tpot: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


async def one_request(client: httpx.AsyncClient, url: str, body: dict) -> Result:
    t0 = time.perf_counter()
    ttft = None
    usage = None
    n_chunks = 0
    try:
        async with client.stream("POST", url, json=body) as resp:
            if resp.status_code != 200:
                return Result(False, error=f"HTTP {resp.status_code}: {(await resp.aread())[:200]!r}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                msg = json.loads(data)
                if msg.get("usage"):
                    usage = msg["usage"]
                for ch in msg.get("choices", []):
                    delta = ch.get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content"):
                        n_chunks += 1
                        if ttft is None:
                            ttft = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        return Result(False, error=repr(e)[:200])
    e2e = time.perf_counter() - t0
    out = (usage or {}).get("completion_tokens", n_chunks)
    if ttft is None:
        ttft = e2e
    tpot = (e2e - ttft) / (out - 1) if out > 1 else 0.0
    return Result(True, ttft, e2e, tpot, (usage or {}).get("prompt_tokens", 0), out)


async def scrape_prefix_cache(client: httpx.AsyncClient, base: str) -> tuple[float, float] | None:
    """(hits, queries) token counters from vLLM's Prometheus endpoint, if exposed."""
    try:
        txt = (await client.get(base.rsplit("/v1", 1)[0] + "/metrics", timeout=10)).text
    except Exception:  # noqa: BLE001
        return None

    def total(name: str) -> float | None:
        vals = [float(m.group(1)) for m in
                re.finditer(rf"^vllm:{name}(?:_total)?(?:{{[^}}]*}})? ([0-9.eE+-]+)$", txt, re.M)]
        return sum(vals) if vals else None

    hits, queries = total("prefix_cache_hits"), total("prefix_cache_queries")
    return (hits, queries) if hits is not None and queries is not None else None


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


async def run(args, concurrency: int) -> dict:
    url = args.base_url.rstrip("/") + "/chat/completions"
    n = args.num_requests or max(16, 4 * concurrency)
    reqs = build_requests(args.workload, n, args.context_words, args.output_tokens, args.model,
                          seed=concurrency)
    limits = httpx.Limits(max_connections=concurrency + 4)
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800), limits=limits) as client:
        # Warm-up (not measured): triggers any remaining compilation; for `prefill`
        # it also places the shared document in the prefix cache when caching is on.
        warm = build_requests(args.workload, 2, args.context_words, args.output_tokens, args.model,
                              seed=999)
        for w in warm:
            await one_request(client, url, w)

        before = await scrape_prefix_cache(client, args.base_url)
        sem = asyncio.Semaphore(concurrency)

        async def bounded(body):
            async with sem:
                return await one_request(client, url, body)

        t0 = time.perf_counter()
        results = await asyncio.gather(*(bounded(r) for r in reqs))
        dur = time.perf_counter() - t0
        after = await scrape_prefix_cache(client, args.base_url)

    ok = [r for r in results if r.ok]
    errors = [r.error for r in results if not r.ok]
    hit_rate = None
    if before and after and after[1] > before[1]:
        hit_rate = (after[0] - before[0]) / (after[1] - before[1])

    def stats(name: str, xs: list[float]) -> dict:
        return {f"{name}_mean_ms": 1000 * statistics.fmean(xs) if xs else 0.0,
                f"{name}_p50_ms": 1000 * pct(xs, 50), f"{name}_p90_ms": 1000 * pct(xs, 90),
                f"{name}_p99_ms": 1000 * pct(xs, 99)}

    out_tok = sum(r.output_tokens for r in ok)
    in_tok = sum(r.prompt_tokens for r in ok)
    row = {
        "tag": args.tag, "workload": args.workload, "concurrency": concurrency,
        "num_requests": n, "completed": len(ok), "failed": len(errors),
        "duration_s": dur,
        "mean_prompt_tokens": in_tok / len(ok) if ok else 0,
        "mean_output_tokens": out_tok / len(ok) if ok else 0,
        "request_throughput_rps": len(ok) / dur,
        "output_throughput_tok_s": out_tok / dur,
        "total_throughput_tok_s": (in_tok + out_tok) / dur,
        "prefix_cache_hit_rate": hit_rate,
        **stats("ttft", [r.ttft for r in ok]),
        **stats("tpot", [r.tpot for r in ok if r.output_tokens > 1]),
        **stats("e2e", [r.e2e for r in ok]),
        "errors_sample": errors[:3],
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma4-e4b")
    ap.add_argument("--workload", choices=["prefill", "unique", "decode"], required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--num-requests", type=int, default=0, help="default: max(16, 4×concurrency)")
    ap.add_argument("--context-words", type=int, default=6000,
                    help="shared document length for prefill/unique (~1.3 tokens per word)")
    ap.add_argument("--output-tokens", type=int, default=None,
                    help="default: 64 for prefill/unique, 512 for decode")
    ap.add_argument("--tag", default="run", help="label, e.g. pc_on / pc_off")
    ap.add_argument("--out", default=str(ROOT / "results" / "results.jsonl"))
    args = ap.parse_args()
    if args.output_tokens is None:
        args.output_tokens = 512 if args.workload == "decode" else 64

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for c in args.concurrency:
        row = asyncio.run(run(args, c))
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        hr = row["prefix_cache_hit_rate"]
        print(f"[{row['tag']}] {row['workload']:<7} c={c:<3} ok={row['completed']}/{row['num_requests']} "
              f"TTFT p50={row['ttft_p50_ms']:.0f}ms  TPOT p50={row['tpot_p50_ms']:.1f}ms  "
              f"out={row['output_throughput_tok_s']:.0f} tok/s  total={row['total_throughput_tok_s']:.0f} tok/s"
              + (f"  cache-hit={hr:.0%}" if hr is not None else ""))
        if row["failed"]:
            print("   errors:", row["errors_sample"])


if __name__ == "__main__":
    main()
