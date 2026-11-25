#!/usr/bin/env bash
# One-time setup on a Cloud TPU v6e VM (Ubuntu 22.04, ct6e-standard-4t).
# Installs Python 3.12 + vLLM-TorchTPU + app/benchmark dependencies.
set -euo pipefail

VENV="${VENV:-$HOME/vllm-venv}"

# 0. tcmalloc (TorchTPU warns about host-side slowdowns without it)
sudo apt-get update -qq && sudo apt-get install -y -qq libtcmalloc-minimal4 || true

# 1. uv (fast installer that can also fetch Python 3.12)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env"
fi

# 2. Python 3.12 venv (vllm-torchtpu wheels target CPython 3.12 only)
uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# 3. vLLM-TorchTPU. Route torch/torchvision to the CPU index so no CUDA wheels
#    are pulled; the plugin brings matching vllm, torch_tpu and libtpu.
export UV_TORCH_BACKEND=cpu
uv pip install vllm-torchtpu

# 4. App + benchmark dependencies
uv pip install -r "$(dirname "$0")/requirements.txt"

# 5. Verify vLLM sees the TPU platform
python -c "from vllm.platforms import current_platform; print('vLLM platform:', current_platform.device_name)"

cat <<EOF

Setup done. Next:
  source $VENV/bin/activate
  hf auth login            # accept the Gemma license on huggingface.co first
  ./serve.sh               # terminal 1
  python -m app.ui         # terminal 2
EOF
