import os
import pathlib

# just set to HF_HOME as LITA_CACHE
# Set LITA_CACHE to the value from the environment or fallback to ~/.cache/lita
if "LITA_CACHE" not in os.environ:
    fallback_path = os.path.join(pathlib.Path.home(), ".cache", "lita")
    os.environ["LITA_CACHE"] = fallback_path
    os.makedirs(fallback_path, exist_ok=True)

os.environ["LITA_ORT_CACHE"] = os.path.join(os.environ["LITA_CACHE"], "onnx")
os.environ["LITA_PROFILE_DIR"] = os.path.join(os.environ["LITA_CACHE"], "lita_uprofile")
os.makedirs(os.environ["LITA_PROFILE_DIR"], exist_ok=True)

os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = "1"

from .lita import Lita

__all__ = [
    "Lita",
]