"""upscale#102 -- target_height is honoured; scale is not silently collapsed.

THE DEFECT. The studio (vivijure-local local-finish/app.ts) sends `target_height: 1080` and
no scale. handler.py never read that key. Sizing was one line:

    final_scale = 4 if int(inp.get("scale", 2) or 2) >= 4 else 2

So a sized request returned 2x the SOURCE, with ok:true. The billed GPU door charged for
the wrong resolution. It only produced 1080 when the source was exactly 540p -- the one
fixture where honoured and ignored are byte-identical. Probe with a non-default source.

WHAT THE FIX HAS TO PROVE, not merely claim:
  1. 720p + target_height 1080 delivers 1920x1080, not 2x=1440.
  2. scale:3 is refused, not collapsed to 2.
  3. An unsatisfiable height (beyond 4x, a downscale, a no-op) is ok:false, not a substitute.
  4. The job paths parse knobs BEFORE any I/O, and thread target_height into _upscale_video.
  5. _upscale_video actually calls _resolve_output_size (a helper nobody calls is decoration).
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


# ---- the decision: _parse_size_knobs --------------------------------------------------------------

def test_neither_knob_defaults_to_scale_2():
    assert handler._parse_size_knobs({}) == (2, None)


def test_target_height_alone_does_not_also_invent_a_scale():
    """The studio sends target_height:1080 and no scale. Defaulting scale to 2 on that
    path is how the two knobs would then disagree on every non-540p source."""
    assert handler._parse_size_knobs({"target_height": 1080}) == (None, 1080)


def test_scale_2_and_scale_4_are_accepted():
    assert handler._parse_size_knobs({"scale": 2}) == (2, None)
    assert handler._parse_size_knobs({"scale": 4}) == (4, None)
    assert handler._parse_size_knobs({"scale": "2"}) == (2, None)


def test_scale_3_is_refused_not_collapsed():
    with pytest.raises(handler.SizeRequestError, match="scale must be 2 or 4, got 3"):
        handler._parse_size_knobs({"scale": 3})


def test_scale_1_and_scale_8_and_scale_true_are_refused():
    for bad in (1, 8, 0, -2, 5, True, 2.5, "three"):
        with pytest.raises(handler.SizeRequestError):
            handler._parse_size_knobs({"scale": bad})


def test_target_height_must_be_an_integer():
    with pytest.raises(handler.SizeRequestError, match="target_height"):
        handler._parse_size_knobs({"target_height": "1080p"})
    with pytest.raises(handler.SizeRequestError, match="target_height"):
        handler._parse_size_knobs({"target_height": True})


# ---- the decision: _resolve_output_size -----------------------------------------------------------

def test_720p_plus_target_1080_is_1920x1080_not_2x():
    """THE HEADLINE. 720 * 2 = 1440. That is the silent-wrong output. Honouring 1080
    is 1920x1080 (1.5x via the GPU interpolate the pipeline already has)."""
    out_w, out_h, scale = handler._resolve_output_size(1280, 720, target_height=1080)
    assert (out_w, out_h) == (1920, 1080), (out_w, out_h)
    assert scale == 1.5


def test_540p_plus_target_1080_is_still_2x():
    """The coincidence the original smoke test could not distinguish: 2x540 = 1080."""
    out_w, out_h, scale = handler._resolve_output_size(960, 540, target_height=1080)
    assert (out_w, out_h) == (1920, 1080)
    assert scale == 2


def test_480p_plus_target_1080_is_2_25x():
    out_w, out_h, scale = handler._resolve_output_size(854, 480, target_height=1080)
    assert out_h == 1080
    assert scale == 2.25


def test_scale_2_of_720p_is_exact_2x():
    out_w, out_h, scale = handler._resolve_output_size(1280, 720, scale=2)
    assert (out_w, out_h, scale) == (2560, 1440, 2)


def test_scale_4_of_tiny_source_is_exact_4x():
    out_w, out_h, scale = handler._resolve_output_size(320, 180, scale=4)
    assert (out_w, out_h, scale) == (1280, 720, 4)


def test_beyond_native_4x_is_refused():
    """240p -> 1080 is 4.5x. The model is 4x native; interpolating UP past that is not
    the super-resolution this door claims."""
    with pytest.raises(handler.SizeRequestError, match="more than 4x"):
        handler._resolve_output_size(426, 240, target_height=1080)


def test_downscale_is_refused():
    with pytest.raises(handler.SizeRequestError, match="does not downscale"):
        handler._resolve_output_size(1920, 1080, target_height=720)


def test_equal_height_is_refused():
    """Already-1080 source + studio target 1080 is a no-op. Refuse so the module
    passthroughs instead of billing a GPU copy."""
    with pytest.raises(handler.SizeRequestError, match="nothing to upscale"):
        handler._resolve_output_size(1920, 1080, target_height=1080)


def test_disagreeing_knobs_are_refused():
    with pytest.raises(handler.SizeRequestError, match="disagree"):
        handler._resolve_output_size(1280, 720, scale=2, target_height=1080)


def test_agreeing_knobs_are_accepted():
    out_w, out_h, scale = handler._resolve_output_size(960, 540, scale=2, target_height=1080)
    assert (out_w, out_h, scale) == (1920, 1080, 2)


def test_target_height_that_hits_the_long_edge_cap_is_refused():
    """A requested height the cap would change is unsatisfiable. Silent 4K substitution
    is the same class of lie as silent 2x."""
    with pytest.raises(handler.SizeRequestError, match="MAX_OUTPUT_LONG_EDGE"):
        handler._resolve_output_size(3840, 2160, target_height=4320, max_edge=3840)


def test_scale_only_long_edge_cap_reports_the_actual_ratio():
    """scale:4 of 1080p is 7680x4320; the cap shrinks it. Report the delivered ratio,
    not the requested 4 -- that is the 'or report the actual delivered size' half."""
    out_w, out_h, scale = handler._resolve_output_size(1920, 1080, scale=4, max_edge=3840)
    assert max(out_w, out_h) == 3840
    assert scale != 4
    assert out_h == 2160
    assert out_w == 3840


def test_odd_target_height_snaps_even_and_reports_actual():
    out_w, out_h, scale = handler._resolve_output_size(1280, 720, target_height=1081)
    assert out_h == 1080
    assert out_w % 2 == 0
    assert scale == 1.5


# ---- job paths parse BEFORE I/O and thread the knobs ---------------------------------------------

R2_JOB = {
    "project": "p",
    "clip_key": "renders/p/clips/s.mp4",
    "output_key": "renders/p/clips/s_up.mp4",
}


class _BoomS3:
    def download_file(self, *a, **k):
        raise AssertionError("must not touch R2")

    def upload_file(self, *a, **k):
        raise AssertionError("must not touch R2")


def test_upscale_r2_refuses_scale_3_before_io(monkeypatch):
    monkeypatch.setattr(handler, "_r2", lambda: _BoomS3())
    out = handler._upscale_r2({**R2_JOB, "scale": 3})
    assert out["ok"] is False
    assert "scale must be 2 or 4" in out["error"]


def test_presigned_path_refuses_scale_3_before_download():
    out = handler.handler({"input": {"video_url": "u", "output_url": "o", "scale": 3}})
    assert out["ok"] is False
    assert "scale must be 2 or 4" in out["error"]


def test_selftest_refuses_scale_3():
    out = handler._selftest({"selftest": True, "scale": 3})
    assert out["ok"] is False
    assert "scale must be 2 or 4" in out["error"]


def test_upscale_r2_threads_target_height_into_upscale_video(monkeypatch):
    seen = {}

    class _S3:
        def download_file(self, *a, **k):
            pass

        def upload_file(self, src, bucket, key, **k):
            pass

    def fake_upscale(model, src, dst, final_scale, budget=None, target_height=None):
        seen["scale"] = final_scale
        seen["target_height"] = target_height
        with open(dst, "wb") as f:
            f.write(b"x")
        return {"frames": 1, "encoder": "libx264", "out_w": 1920, "out_h": 1080, "scale": 1.5}

    monkeypatch.setattr(handler, "_r2", lambda: _S3())
    monkeypatch.setattr(handler, "_load_model", lambda name: object())
    monkeypatch.setattr(handler, "_upscale_video", fake_upscale)
    out = handler._upscale_r2({**R2_JOB, "target_height": 1080})
    assert seen == {"scale": None, "target_height": 1080}, seen
    assert out["ok"] is True
    assert out["out_w"] == 1920 and out["out_h"] == 1080
    assert out["scale"] == 1.5
    assert out["applied"] == ["upscale:1080h"]


def test_upscale_r2_scale_2_still_tags_2x(monkeypatch):
    class _S3:
        def download_file(self, *a, **k):
            pass

        def upload_file(self, src, bucket, key, **k):
            pass

    def fake_upscale(model, src, dst, final_scale, budget=None, target_height=None):
        with open(dst, "wb") as f:
            f.write(b"x")
        return {"frames": 1, "encoder": "libx264", "out_w": 2560, "out_h": 1440, "scale": 2}

    monkeypatch.setattr(handler, "_r2", lambda: _S3())
    monkeypatch.setattr(handler, "_load_model", lambda name: object())
    monkeypatch.setattr(handler, "_upscale_video", fake_upscale)
    out = handler._upscale_r2({**R2_JOB, "scale": 2})
    assert out["ok"] is True
    assert out["applied"] == ["upscale:2x"]
    assert out["scale"] == 2


# ---- _upscale_video is wired to the resolver (a helper nobody calls is decoration) ----------------

def test_upscale_video_calls_resolve_output_size_with_target_height(monkeypatch):
    called = {}

    def fake_resolve(sw, sh, scale=None, target_height=None, max_edge=None):
        called["args"] = (sw, sh, scale, target_height)
        return 1920, 1080, 1.5

    captured = {}

    def fake_batch(model, frames_np, out_w, out_h, start_tile=None):
        captured["size"] = (out_w, out_h)
        return [b"o" for _ in frames_np], 512

    class _Pipe:
        def __init__(self):
            self._left = 2

        def read(self, n):
            if self._left <= 0:
                return b""
            self._left -= 1
            return b"\x00" * n

        def close(self):
            pass

    class _Enc:
        def __init__(self):
            self.stdin = types.SimpleNamespace(write=lambda b: None, close=lambda: None)

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    def fake_popen(cmd, **kw):
        if "rawvideo" in cmd and cmd[-1] == "-":
            return types.SimpleNamespace(stdout=_Pipe(), wait=lambda timeout=None: 0, kill=lambda: None)
        return _Enc()

    handler.np.frombuffer = lambda b, dtype=None: types.SimpleNamespace(reshape=lambda *a: b)
    handler.np.ascontiguousarray = lambda x: types.SimpleNamespace(tobytes=lambda: x)
    handler.np.uint8 = "u8"
    monkeypatch.setattr(handler, "_resolve_output_size", fake_resolve)
    monkeypatch.setattr(handler, "_ffprobe",
                        lambda src, spec, **kw: (["24/1"] if "frame_rate" in spec else ["1280", "720"]))
    monkeypatch.setattr(handler, "_nvenc_available", lambda: False)
    monkeypatch.setattr(handler, "_has_audio", lambda src, **kw: False)
    monkeypatch.setattr(handler, "_upscale_batch", fake_batch)
    monkeypatch.setattr(handler.subprocess, "Popen", fake_popen)

    info = handler._upscale_video(object(), "in.mp4", "out.mp4", None, target_height=1080)
    assert called["args"] == (1280, 720, None, 1080), called
    assert captured["size"] == (1920, 1080), captured
    assert info["out_w"] == 1920 and info["out_h"] == 1080
    assert info["scale"] == 1.5


def test_collapse_assignment_is_gone_from_the_source():
    """The live-container measurement that opened #102. A docstring that QUOTES the
    old line is not the assignment; match the assignment form so a mention is not a
    false red."""
    src = open(handler.__file__, encoding="utf-8").read()
    assert "target_height" in src
    assert "final_scale = 4 if int(" not in src
    assert src.count("_parse_size_knobs") >= 4  # definition + r2 + presigned + selftest
    assert src.count("_resolve_output_size") >= 2  # definition + _upscale_video
