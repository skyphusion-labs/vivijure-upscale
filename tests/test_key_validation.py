"""Project-scoped R2 key gate (no GPU / network)."""

import os
import sys
import types


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


_stub("torch", __version__="0-stub", inference_mode=lambda *a, **k: (lambda f: f),
      cuda=types.SimpleNamespace(is_available=lambda: False))
_stub("boto3", client=lambda *a, **k: None)
_stub("numpy")
_stub("requests")
_stub("spandrel", ModelLoader=object)
_runpod = _stub("runpod")
_runpod.serverless = types.SimpleNamespace(start=lambda *a, **k: None)

os.environ.setdefault("R2_ENDPOINT_URL", "https://stub.r2")
os.environ.setdefault("R2_ACCESS_KEY_ID", "stub")
os.environ.setdefault("R2_SECRET_ACCESS_KEY", "stub")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import handler  # noqa: E402


def test_scoped_rejects_cross_project():
    err = handler._scoped_key_error(
        "renders/victim/clips/s.mp4", "clip_key", project="attacker")
    assert err and "must be under renders/attacker/" in err


def test_scoped_rejects_missing_project():
    err = handler._scoped_key_error("renders/p/clips/s.mp4", "clip_key", project="")
    assert err and "project is required" in err


def test_scoped_accepts_matching_project():
    assert handler._scoped_key_error(
        "renders/neon/clips/s.mp4", "clip_key", project="neon") is None


def test_upscale_r2_refuses_cross_project_before_io(monkeypatch):
    class Boom:
        def download_file(self, *a, **k):
            raise AssertionError("must not touch R2")

        def upload_file(self, *a, **k):
            raise AssertionError("must not touch R2")

    monkeypatch.setattr(handler, "_r2", lambda: Boom())
    out = handler._upscale_r2({
        "project": "attacker",
        "clip_key": "renders/victim/clips/s.mp4",
    })
    assert out["ok"] is False
    assert "must be under renders/attacker/" in out["error"]
