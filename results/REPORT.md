# Gemma 4 E4B-it on 1× TPU v6e (vLLM-TorchTPU) — serving benchmark

TTFT = time to first token · TPOT = time per output token after the first · throughput = tokens/s across all concurrent requests. p50 unless noted.

## Prefill-heavy · shared long document + short question · up to 64 output tokens

| Concurrency | Prefix cache | Prompt tok | TTFT p50 (ms) | TTFT p90 (ms) | TPOT p50 (ms) | E2E p50 (ms) | Req/s | Output tok/s | Total tok/s | Cache hit |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | off | 7,081 | 170 | 170 | 9.1 | 324 | 2.90 | 58 | 20,602 | — |
| 1 | on | 7,081 | 35 | 36 | 9.2 | 192 | 4.72 | 95 | 33,538 | 99% |
| 4 | off | 7,081 | 183 | 466 | 28.0 | 766 | 4.79 | 96 | 33,985 | — |
| 4 | on | 7,081 | 49 | 866 | 11.0 | 261 | 6.46 | 130 | 45,853 | 99% |
| 16 | off | 7,081 | 601 | 1,838 | 103.4 | 2,845 | 5.08 | 101 | 36,048 | — |
| 16 | on | 7,081 | 83 | 1,565 | 18.3 | 452 | 16.23 | 324 | 115,238 | 99% |

**Prefix caching on vs off**

- c=1: TTFT 4.8× lower with caching; total throughput 1.63×
- c=4: TTFT 3.7× lower with caching; total throughput 1.35×
- c=16: TTFT 7.3× lower with caching; total throughput 3.20×

## Prefill-heavy control · unique preamble per request (cache cannot hit)

| Concurrency | Prefix cache | Prompt tok | TTFT p50 (ms) | TTFT p90 (ms) | TPOT p50 (ms) | E2E p50 (ms) | Req/s | Output tok/s | Total tok/s | Cache hit |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | off | 7,119 | 170 | 170 | 9.1 | 325 | 2.90 | 58 | 20,675 | — |
| 1 | on | 7,118 | 171 | 171 | 9.1 | 325 | 2.90 | 58 | 20,671 | 0% |
| 4 | off | 7,118 | 183 | 400 | 27.0 | 728 | 4.85 | 98 | 34,629 | — |
| 4 | on | 7,119 | 185 | 469 | 28.1 | 747 | 4.76 | 96 | 33,967 | 0% |
| 16 | off | 7,119 | 468 | 1,478 | 107.3 | 2,615 | 5.78 | 116 | 41,263 | — |
| 16 | on | 7,119 | 475 | 1,572 | 113.5 | 2,764 | 5.11 | 102 | 36,465 | 0% |

**Prefix caching on vs off**

- c=1: TTFT 1.0× lower with caching; total throughput 1.00×
- c=4: TTFT 1.0× lower with caching; total throughput 0.98×
- c=16: TTFT 1.0× lower with caching; total throughput 0.88×

## Decode-heavy · short prompt · 512 output tokens (EOS ignored)

| Concurrency | Prefix cache | Prompt tok | TTFT p50 (ms) | TTFT p90 (ms) | TPOT p50 (ms) | E2E p50 (ms) | Req/s | Output tok/s | Total tok/s | Cache hit |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | off | 30 | 16 | 17 | 8.9 | 4,567 | 0.22 | 112 | 119 | — |
| 1 | on | 30 | 16 | 17 | 8.9 | 4,566 | 0.22 | 112 | 119 | 0% |
| 4 | off | 30 | 77 | 1,139 | 9.0 | 4,970 | 0.79 | 402 | 426 | — |
| 4 | on | 30 | 32 | 94 | 9.0 | 4,667 | 0.86 | 438 | 464 | 0% |
| 16 | off | 30 | 179 | 3,420 | 11.4 | 6,655 | 2.23 | 1,142 | 1,209 | — |
| 16 | on | 30 | 78 | 3,397 | 11.3 | 6,661 | 2.26 | 1,157 | 1,225 | 0% |

**Prefix caching on vs off**

- c=1: TTFT 1.0× lower with caching; total throughput 1.00×
- c=4: TTFT 2.4× lower with caching; total throughput 1.09×
- c=16: TTFT 2.3× lower with caching; total throughput 1.01×

