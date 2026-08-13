import io
import os
import sys
import types
from pathlib import Path


class _Secrets:
    def get(self, key, default=None):
        return os.environ.get(str(key).upper(), default)

    def __getitem__(self, key):
        return os.environ[str(key).upper()]


def _cache(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def decorator(function):
        return function

    decorator.clear = lambda: None
    return decorator


class _StreamlitShim(types.ModuleType):
    secrets = _Secrets()
    cache_data = staticmethod(_cache)
    cache_resource = staticmethod(_cache)

    def __getattr__(self, name):
        if name in {"warning", "error", "info", "caption", "success", "write", "toast", "exception"}:
            return lambda *args, **kwargs: print(f"[streamlit.{name}]", *args, flush=True)
        return lambda *args, **kwargs: None


ROOT_DIR = Path(__file__).resolve().parents[2]
APP_PATH = ROOT_DIR / "etf_app.py"
UI_MARKER = "# ─── Streamlit UI"

sys.modules["streamlit"] = _StreamlitShim("streamlit")

source = APP_PATH.read_text(encoding="utf-8")
cut = source.find(UI_MARKER)
if cut == -1:
    raise RuntimeError(f"Cannot find Streamlit UI marker in {APP_PATH}")

core = types.ModuleType("etf_core")
core.__file__ = str(APP_PATH)
exec(compile(source[:cut], str(APP_PATH), "exec"), core.__dict__)


class UploadBuffer(io.BytesIO):
    def __init__(self, content: bytes, filename: str):
        super().__init__(content)
        self.name = filename
