"""upscale#126: a refusal and a crash carry different error_type values.

The class is chosen at the return. It is not parsed from the English. The soft-degrade
arm keeps `detail` and does not gain a failure class. A spelling outside the closed set
raises, so the set cannot silently become a free string.
"""

import os
import sys
import types

import pytest


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


_stub("torch", __version__="0-stub", inference_mode=lambda *a, **k: (lambda f: f),
      cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None,
                                 synchronize=lambda: None))
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


R2_JOB = {"input": {"project": "p", "clip_key": "renders/p/a.mp4"}}


def test_closed_set_rejects_a_near_miss_and_a_non_member():
    with pytest.raises(ValueError):
        handler._failure("not_configured", "x")
    with pytest.raises(ValueError):
        handler._failure("quota", "x")


def test_a_missing_r2_config_is_not_configured(monkeypatch):
    monkeypatch.setattr(handler, "R2_ENDPOINT", "")
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    out = handler.handler(R2_JOB)
    assert out["ok"] is False
    assert out["error_type"] == "not-configured"
    assert "R2_ENDPOINT_URL" in out["error"]


def test_a_crash_is_internal_and_distinct_from_the_refusal(monkeypatch):
    monkeypatch.setattr(handler, "R2_ENDPOINT", "https://stub.r2")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "stub")

    def boom():
        raise RuntimeError("disk fell off")

    monkeypatch.setattr(handler, "_r2", boom)
    out = handler.handler(R2_JOB)
    assert out["error_type"] == "internal"
    assert out["error_type"] != "not-configured"
    assert "disk fell off" in out["error"]


def test_a_presigned_request_with_no_urls_is_bad_input():
    out = handler.handler({"input": {}})
    assert out["error_type"] == "bad-input"
    assert "video_url" in out["error"]


def test_the_degrade_arm_keeps_detail_and_omits_error_type(monkeypatch):
    monkeypatch.setattr(handler, "R2_ENDPOINT", "https://stub.r2")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "stub")

    class S3:
        def download_file(self, *_a, **_k):
            raise handler.InvocationExpired("budget")

    monkeypatch.setattr(handler, "_r2", lambda: S3())
    out = handler.handler(R2_JOB)
    assert out["ok"] is False
    assert "detail" in out
    assert "error" not in out
    assert "error_type" not in out


def test_empty_upscale_output_is_backend_error(monkeypatch):
    monkeypatch.setattr(handler, "R2_ENDPOINT", "https://stub.r2")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "stub")

    class S3:
        def download_file(self, *_a, **_k):
            return None

        def upload_file(self, *_a, **_k):
            raise AssertionError("upload must not run when there is no output")

    monkeypatch.setattr(handler, "_r2", lambda: S3())
    monkeypatch.setattr(handler, "_load_model", lambda _name: object())
    monkeypatch.setattr(
        handler, "_upscale_video",
        lambda *_a, **_k: {"scale": 2, "out_w": 2, "out_h": 2, "frames": 1, "encoder": "x"},
    )
    monkeypatch.setattr(handler.os.path, "getsize", lambda _p: 0)
    out = handler.handler(R2_JOB)
    assert out["error_type"] == "backend-error"
    assert out["error"] == "upscale produced no output"
