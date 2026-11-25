#!/usr/bin/env bash
# Full benchmark matrix: prefix caching {on, off} × workloads {prefill, unique, decode}
# × concurrency {1, 4, 16}. Restarts vLLM for each caching setting, then writes
# results/results.jsonl and results/REPORT.md.
#
#   ./bench/run_matrix.sh                      # defaults
#   CONCURRENCY="1 8 32" ./bench/run_matrix.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CONCURRENCY="${CONCURRENCY:-1 4 16}"
PORT="${PORT:-8000}"
OUT="results/results.jsonl"
mkdir -p results logs
[[ -f samples/service_agreement.pdf ]] || python scripts/make_samples.py

wait_ready() {
  echo -n ">> waiting for vLLM (first start compiles TPU graphs; can take several minutes)"
  for _ in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then echo " ready"; return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo; echo "!! server exited, see $LOG"; exit 1; fi
    echo -n "."; sleep 10
  done
  echo; echo "!! timed out waiting for server"; exit 1
}

for PC in on off; do
  LOG="logs/serve_pc_${PC}.log"
  echo "================ prefix caching: $PC ================"
  PREFIX_CACHING=$PC PORT=$PORT ./serve.sh >"$LOG" 2>&1 &
  SERVER_PID=$!
  trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
  wait_ready

  for WL in prefill unique decode; do
    # shellcheck disable=SC2086
    python -m bench.bench --base-url "http://127.0.0.1:$PORT/v1" --workload "$WL" \
      --concurrency $CONCURRENCY --tag "pc_${PC}" --out "$OUT"
  done

  kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true
  sleep 5
done

python -m bench.report "$OUT" > results/REPORT.md
echo ">> wrote results/REPORT.md"
cat results/REPORT.md
