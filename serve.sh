#!/usr/bin/env bash
# Serve Gemma 4 E4B-it on ONE v6e chip with vLLM-TorchTPU (OpenAI-compatible API on :8000).
#
#   ./serve.sh                     # prefix caching ON (default)
#   PREFIX_CACHING=off ./serve.sh  # prefix caching OFF (for the benchmark matrix)
#
# Other knobs (env vars): MODEL, PORT, MAX_MODEL_LEN, MAX_NUM_SEQS, MAX_BATCHED_TOKENS, CHIP
set -euo pipefail

MODEL="${MODEL:-google/gemma-4-E4B-it}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
PREFIX_CACHING="${PREFIX_CACHING:-on}"
CHIP="${CHIP:-0}"

# --- tcmalloc speeds up the TorchTPU host runtime (installed by setup.sh) ---
TCMALLOC=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
[[ -f "$TCMALLOC" ]] && export LD_PRELOAD="$TCMALLOC"

# --- Pin the server to a single chip of the 4-chip ct6e-standard-4t host ---
# Standard libtpu env vars: one process, one chip.
export TPU_VISIBLE_CHIPS="$CHIP"
export TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1"
export TPU_PROCESS_BOUNDS="1,1,1"

# --- Reuse compiled XLA/Pallas graphs across restarts (first start compiles for minutes) ---
export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-$HOME/.cache/vllm_xla}"
mkdir -p "$VLLM_XLA_CACHE_PATH"

if [[ "$PREFIX_CACHING" == "on" ]]; then
  PC_FLAG="--enable-prefix-caching"
else
  PC_FLAG="--no-enable-prefix-caching"
fi

echo ">> Serving $MODEL on chip $CHIP, port $PORT, prefix caching: $PREFIX_CACHING"

exec vllm serve "$MODEL" \
  --served-model-name gemma4-e4b \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --limit-mm-per-prompt '{"image": 4, "audio": 0}' \
  --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
  $PC_FLAG \
  --host "${HOST:-127.0.0.1}" \
  --port "$PORT"
