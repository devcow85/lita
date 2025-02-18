import os
import pathlib

# Set LITA_CACHE to the value from the environment or fallback to ~/.cache/lita
lita_cache = os.environ.get("LITA_CACHE") or os.path.join(pathlib.Path.home(), ".cache", "lita")
os.environ["LITA_CACHE"] = lita_cache
os.environ["LITA_ONNX_CACHE"] = os.path.join(lita_cache, "onnx")
os.environ["TRANSFORMERS_CACHE"] = lita_cache   # Hf_HOME not working
# os.environ["HF_HOME"] = lita_cache   # Hf_HOME not working
os.environ["VLLM_CACHE_ROOT"] = os.path.join(lita_cache, 'vllm')

os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = "1"

from .lita import Lita

__all__ = [
    "Lita",
]