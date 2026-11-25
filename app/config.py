"""Runtime settings, overridable with environment variables."""
import os

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("VLLM_MODEL", "gemma4-e4b")  # --served-model-name in serve.sh
API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

# Rendering / multimodal limits (serve.sh allows 4 images per prompt)
RENDER_DPI = int(os.environ.get("RENDER_DPI", "150"))
MAX_IMAGE_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1600"))
MAX_PAGES = int(os.environ.get("MAX_PAGES", "40"))
MAX_IMAGES_PER_REQUEST = int(os.environ.get("MAX_IMAGES_PER_REQUEST", "4"))

# A page whose text layer has fewer characters than this is treated as scanned
SCANNED_TEXT_THRESHOLD = int(os.environ.get("SCANNED_TEXT_THRESHOLD", "40"))

# Retrieval
CHUNK_WORDS = int(os.environ.get("CHUNK_WORDS", "120"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "30"))
TOP_K = int(os.environ.get("TOP_K", "5"))

# Gemma 4 has an optional thinking mode; keep it off for low-latency extraction.
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "0") == "1"
