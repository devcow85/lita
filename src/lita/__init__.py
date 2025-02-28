import os
import pathlib

# just set to HF_HOME as LITA_CACHE
# Set LITA_CACHE to the value from the environment or fallback to ~/.cache/lita
lita_cache = os.environ.get("LITA_CACHE") or os.path.join(pathlib.Path.home(), ".cache", "lita")

os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = "1"

from .lita import Lita

__all__ = [
    "Lita",
]