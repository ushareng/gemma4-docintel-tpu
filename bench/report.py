"""Turn results.jsonl into a Markdown report comparing prefix caching on vs off.

  python -m bench.report results/results.jsonl > results/REPORT.md
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

WORKLOAD_NOTES = {
    "prefill": "Prefill-heavy · shared long document + short question · up to 64 output tokens",
    "unique": "Prefill-heavy control · unique preamble per request (cache cannot hit)",
    "decode": "Decode-heavy · short prompt · 512 output tokens (EOS ignored)",
}


def load(path: str) -> list[dict]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    latest: dict[tuple, dict] = {}
    for r in rows:  # keep the last run of each (tag, workload, concurrency)
        latest[(r["tag"], r["workload"], r["concurrency"])] = r
    return list(latest.values())


def fmt(v, nd=0):
    return "—" if v is None else f"{v:,.{nd}f}"


def main(path: str) -> None:
    rows = load(path)
    by = defaultdict(dict)
    for r in rows:
        by[(r["workload"], r["concurrency"])][r["tag"]] = r

    print("# Gemma 4 E4B-it on 1× TPU v6e (vLLM-TorchTPU) — serving benchmark\n")
    print("TTFT = time to first token · TPOT = time per output token after the first · "
          "throughput = tokens/s across all concurrent requests. p50 unless noted.\n")

    for wl in ["prefill", "unique", "decode"]:
        keys = sorted(c for (w, c) in by if w == wl)
        if not keys:
            continue
        print(f"## {WORKLOAD_NOTES[wl]}\n")
        print("| Concurrency | Prefix cache | Prompt tok | TTFT p50 (ms) | TTFT p90 (ms) | "
              "TPOT p50 (ms) | E2E p50 (ms) | Req/s | Output tok/s | Total tok/s | Cache hit |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for c in keys:
            for tag in sorted(by[(wl, c)]):
                r = by[(wl, c)][tag]
                hr = r.get("prefix_cache_hit_rate")
                print(f"| {c} | {tag.replace('pc_', '')} | {fmt(r['mean_prompt_tokens'])} | "
                      f"{fmt(r['ttft_p50_ms'])} | {fmt(r['ttft_p90_ms'])} | {fmt(r['tpot_p50_ms'], 1)} | "
                      f"{fmt(r['e2e_p50_ms'])} | {fmt(r['request_throughput_rps'], 2)} | "
                      f"{fmt(r['output_throughput_tok_s'])} | {fmt(r['total_throughput_tok_s'])} | "
                      f"{'—' if hr is None else f'{hr:.0%}'} |")
        print()

        # Headline speedups (caching on vs off)
        lines = []
        for c in keys:
            on, off = by[(wl, c)].get("pc_on"), by[(wl, c)].get("pc_off")
            if on and off and on["ttft_p50_ms"] and off["ttft_p50_ms"]:
                lines.append(f"- c={c}: TTFT {off['ttft_p50_ms'] / on['ttft_p50_ms']:.1f}× lower with "
                             f"caching; total throughput {on['total_throughput_tok_s'] / max(off['total_throughput_tok_s'], 1e-9):.2f}×")
        if lines:
            print("**Prefix caching on vs off**\n")
            print("\n".join(lines) + "\n")

    failed = [r for r in rows if r.get("failed")]
    if failed:
        print("## Failures\n")
        for r in failed:
            print(f"- {r['tag']} {r['workload']} c={r['concurrency']}: {r['failed']} failed — {r['errors_sample'][:1]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/results.jsonl")
