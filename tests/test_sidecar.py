"""Unit coverage for the #583 provenance-sidecar stamp (no GPU needed).

Per the ruled design the studio CORE computes the param-hash and passes it as an opaque `output_hash`;
this endpoint STAMPS it to `<output_key>.hash` AFTER the artifact (artifact first, sidecar last -- the
only safe order). The heavy GPU/ML/network deps import at module load, so they are STUBBED here; the
stamp under test is pure control flow and the tests monkeypatch _load_model / _upscale_video / _r2.
"""

import os
import sys
import types


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


_stub("torch", __version__="0-stub", inference_mode=lambda *a, **k: (lambda f: f))
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


class _FakeS3:
    def __init__(self):
        self.order = []           # ("artifact"|"sidecar", key), to assert ordering
        self.puts = []            # (Key, Body) put_object calls -> the .hash sidecar

    def download_file(self, bucket, key, dst):
        open(dst, "wb").close()

    def upload_file(self, src, bucket, key, **k):
        self.order.append(("artifact", key))

    def put_object(self, Bucket=None, Key=None, Body=None, **k):
        self.puts.append((Key, Body))
        self.order.append(("sidecar", Key))


def _run_ok(model, src, dst, final_scale):
    with open(dst, "wb") as f:
        f.write(b"video-bytes")
    return {"frames": 10, "encoder": "libx264"}


R2_JOB = {
    "project": "p",
    "clip_key": "renders/p/clips/s.mp4",
    "output_key": "renders/p/clips/s_up.mp4",
}


def _wire(monkeypatch, s3):
    monkeypatch.setattr(handler, "_r2", lambda: s3)
    monkeypatch.setattr(handler, "_load_model", lambda name: object())
    monkeypatch.setattr(handler, "_upscale_video", _run_ok)


def test_r2_stamps_sidecar_after_artifact_when_output_hash_present(monkeypatch):
    s3 = _FakeS3()
    _wire(monkeypatch, s3)
    out = handler._upscale_r2({**R2_JOB, "output_hash": "deadbeef"})
    assert out["ok"] is True
    assert s3.puts == [("renders/p/clips/s_up.mp4.hash", b"deadbeef")]  # verbatim value
    assert [kind for kind, _ in s3.order] == ["artifact", "sidecar"]    # artifact FIRST


def test_r2_writes_no_sidecar_without_output_hash(monkeypatch):
    s3 = _FakeS3()
    _wire(monkeypatch, s3)
    out = handler._upscale_r2(dict(R2_JOB))
    assert out["ok"] is True
    assert s3.puts == []  # legacy core -> no sidecar, safe re-run at the gate


def test_r2_sidecar_failure_never_fails_the_render(monkeypatch):
    class _S3Boom(_FakeS3):
        def put_object(self, **k):
            raise RuntimeError("r2 down")
    s3 = _S3Boom()
    _wire(monkeypatch, s3)
    out = handler._upscale_r2({**R2_JOB, "output_hash": "deadbeef"})
    assert out["ok"] is True and "error" not in out  # artifact up; sidecar miss is best-effort
