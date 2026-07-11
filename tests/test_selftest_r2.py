"""Unit coverage for the #26 R2 selftest leg (_selftest_r2) and its aggregation into _selftest.

The R2 leg exercises the REAL finish contract (_upscale_r2 boto3 download+upload). HONEST-FAILURES rule:
absent R2 creds -> the leg reports itself explicitly skipped (ok None) and does NOT fail the sweep, UNLESS
the caller asked for it (`requested`), in which case absent creds are a FAILURE. The heavy GPU/ML/network
deps import at module load, so they are STUBBED; the leg boto3 + subprocess calls are monkeypatched, so
this is control-flow coverage with no GPU and no network.
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


_stub("torch", __version__="0-stub", inference_mode=lambda *a, **k: (lambda f: f),
      cuda=types.SimpleNamespace(is_available=lambda: False))
_stub("boto3", client=lambda *a, **k: None)
_stub("numpy")
_stub("requests")
_stub("spandrel", ModelLoader=object)
_runpod = _stub("runpod")
_runpod.serverless = types.SimpleNamespace(start=lambda *a, **k: None)

# Seed R2 env BEFORE importing handler so the R2_* module globals (bound at import) are truthy. This also
# keeps handler-import order-independent across the test files (test_sidecar relies on the same seed); each
# test overrides creds via monkeypatch. Match the values test_sidecar uses.
os.environ.setdefault("R2_ENDPOINT_URL", "https://stub.r2")
os.environ.setdefault("R2_ACCESS_KEY_ID", "stub")
os.environ.setdefault("R2_SECRET_ACCESS_KEY", "stub")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import handler  # noqa: E402


class _FakeS3:
    def __init__(self):
        self.uploaded = []
        self.deleted = []

    def upload_file(self, src, bucket, key, **k):
        self.uploaded.append(key)

    def head_object(self, Bucket=None, Key=None, **k):
        return {"ContentLength": 4242}

    def delete_object(self, Bucket=None, Key=None, **k):
        self.deleted.append(Key)


def _wire_creds(monkeypatch):
    monkeypatch.setattr(handler, "R2_ENDPOINT", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "stub")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "stub")


def _wire_no_creds(monkeypatch):
    monkeypatch.setattr(handler, "R2_ENDPOINT", "")
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)


def _fake_gen_ok(*a, **k):
    return types.SimpleNamespace(returncode=0, stderr="")


# ---- honest-skip vs required-failure when creds are absent ----------------------------------------

def test_leg_skips_when_no_creds_and_not_requested(monkeypatch):
    _wire_no_creds(monkeypatch)
    leg = handler._selftest_r2(2, "realesr-animevideov3", requested=False)
    assert leg == {"ok": None, "skipped": "no creds"}   # explicit skip, never a silent green


def test_leg_fails_when_no_creds_but_requested(monkeypatch):
    _wire_no_creds(monkeypatch)
    leg = handler._selftest_r2(2, "realesr-animevideov3", requested=True)
    assert leg["ok"] is False
    assert "requested" in leg and leg["requested"] is True
    assert "creds" in leg["error"].lower() or "set" in leg["error"].lower()


# ---- the real round-trip (mocked s3 + _upscale_r2) -------------------------------------------------

def test_leg_round_trips_and_cleans_up(monkeypatch):
    _wire_creds(monkeypatch)
    s3 = _FakeS3()
    monkeypatch.setattr(handler, "_r2", lambda: s3)
    monkeypatch.setattr(handler.subprocess, "run", _fake_gen_ok)
    monkeypatch.setattr(handler, "_upscale_r2",
                        lambda inp: {"ok": True, "encoder": "h264_nvenc", "frames": 12,
                                     "clip_key": inp["output_key"]})
    leg = handler._selftest_r2(4, "realesr-animevideov3", requested=True)
    assert leg["ok"] is True
    assert leg["output_bytes"] == 4242
    assert leg["encoder"] == "h264_nvenc" and leg["frames"] == 12
    ck, ok_ = leg["clip_key"], leg["output_key"]
    assert ck.startswith("renders/_selftest/") and ok_.endswith("_up.mp4")
    assert s3.uploaded == [ck]
    # both objects AND the provenance sidecar are deleted after (best-effort cleanup)
    assert set(s3.deleted) == {ck, ok_, f"{ok_}.hash"}


def test_leg_reports_upscale_failure_and_still_cleans_up(monkeypatch):
    _wire_creds(monkeypatch)
    s3 = _FakeS3()
    monkeypatch.setattr(handler, "_r2", lambda: s3)
    monkeypatch.setattr(handler.subprocess, "run", _fake_gen_ok)
    monkeypatch.setattr(handler, "_upscale_r2", lambda inp: {"ok": False, "error": "boom"})
    leg = handler._selftest_r2(2, "realesr-animevideov3", requested=True)
    assert leg["ok"] is False and leg["error"] == "boom"
    assert len(s3.deleted) == 3   # cleanup ran even on the failure path


# ---- aggregation into the sweep -------------------------------------------------------------------

def test_sweep_ok_when_models_pass_and_r2_skips(monkeypatch):
    monkeypatch.setattr(handler, "_selftest_one", lambda name, scale, *a, **k: {"ok": True, "model": name})
    monkeypatch.setattr(handler, "_selftest_r2", lambda *a, **k: {"ok": None, "skipped": "no creds"})
    res = handler._selftest({"selftest": True})
    assert res["ok"] is True                     # a skipped R2 leg does NOT fail the sweep
    assert res["r2"]["skipped"] == "no creds"


def test_sweep_fails_when_r2_leg_fails(monkeypatch):
    monkeypatch.setattr(handler, "_selftest_one", lambda name, scale, *a, **k: {"ok": True})
    monkeypatch.setattr(handler, "_selftest_r2", lambda *a, **k: {"ok": False, "error": "r2 down"})
    res = handler._selftest({"selftest": True, "r2": True})
    assert res["ok"] is False                    # a FAILED R2 leg fails the sweep


def test_sweep_fails_when_a_model_fails_even_if_r2_ok(monkeypatch):
    monkeypatch.setattr(handler, "_selftest_one",
                        lambda name, scale, *a, **k: {"ok": name != "RealESRGAN_x4plus"})
    monkeypatch.setattr(handler, "_selftest_r2", lambda *a, **k: {"ok": True})
    res = handler._selftest({"selftest": True})
    assert res["ok"] is False


def test_single_model_with_r2_flag_attaches_leg(monkeypatch):
    monkeypatch.setattr(handler, "_selftest_one", lambda name, scale, *a, **k: {"ok": True, "model": name})
    monkeypatch.setattr(handler, "_selftest_r2", lambda *a, **k: {"ok": True, "model": "x"})
    res = handler._selftest({"selftest": True, "model": "RealESRGAN_x4plus", "r2": True})
    assert res["ok"] is True and res["r2"]["ok"] is True


def test_res_and_dur_thread_to_selftest_one(monkeypatch):
    seen = []
    monkeypatch.setattr(handler, "_selftest_one",
                        lambda name, scale, res="1280x720", dur=3: seen.append((name, scale, res, dur))
                        or {"ok": True})
    monkeypatch.setattr(handler, "_selftest_r2", lambda *a, **k: {"ok": None, "skipped": "no creds"})
    handler._selftest({"selftest": True, "scale": 4, "res": "2560x1440", "dur": 1})
    assert seen and all(r == "2560x1440" and d == 1 and sc == 4 for _, sc, r, d in seen)
