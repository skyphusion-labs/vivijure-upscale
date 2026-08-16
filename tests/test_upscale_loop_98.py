"""upscale#98 -- abort-path leak, per-batch tile re-search, and the remaining silent-wrong-output.

The first two defects live in `_upscale_video`'s batch loop and shipped in #99. The remaining one
is the same loop: a long RealESRGAN_x4plus shot burns the whole wall-clock guard, then the job
path returns ok:false and the module passthroughs, so the film ships un-upscaled. SCENE_MAX_SECONDS
is 60; the door admits ~10s of 720p x4plus inside a 1200s budget. Raising the budget just hits
PHASE_HARD_DEADLINE_SECONDS (5400) instead.

The honest leftover: after the first settled batch, project remaining work against the remaining
budget and refuse NOW with InvocationExpired. Same {ok:false, detail} the module already treats
as degrade. Never ok:true at the source resolution -- that is the billed lie (#102's sibling).

Heavy deps are stubbed (no GPU, no ffmpeg). The control flow under test is the shipped one.
"""
import contextlib
import os
import sys
import time
import types

import pytest


def _load_handler(env=None):
    """Import handler.py with the heavy deps stubbed. `env` is applied BEFORE import because the
    module reads its tunables at import time (which is also why FFMPEG_TIMEOUT cannot be changed
    within a container's life)."""
    for k, v in (env or {}).items():
        os.environ[k] = str(v)

    calls = {"empty_cache": 0, "synchronize": 0}
    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = "0-stub"
    torch_mod.float16 = "f16"
    torch_mod.inference_mode = lambda *a, **k: (lambda f: f)
    torch_mod.autocast = lambda **k: contextlib.nullcontext()
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: calls.__setitem__("empty_cache", calls["empty_cache"] + 1),
        synchronize=lambda: calls.__setitem__("synchronize", calls["synchronize"] + 1),
    )

    np_mod = types.ModuleType("numpy")
    np_mod.uint8 = "u8"
    np_mod.frombuffer = lambda b, dtype=None: types.SimpleNamespace(reshape=lambda *a: b)
    np_mod.ascontiguousarray = lambda x: types.SimpleNamespace(tobytes=lambda: x)

    for name, mod in {
        "torch": torch_mod,
        "numpy": np_mod,
        "boto3": types.SimpleNamespace(client=lambda *a, **k: None),
        "requests": types.ModuleType("requests"),
        "spandrel": types.SimpleNamespace(ModelLoader=object),
    }.items():
        if isinstance(mod, types.ModuleType):
            sys.modules[name] = mod
        else:
            m = types.ModuleType(name)
            for k in dir(mod):
                if not k.startswith("__"):
                    setattr(m, k, getattr(mod, k))
            sys.modules[name] = m
    runpod = types.ModuleType("runpod")
    runpod.serverless = types.SimpleNamespace(start=lambda *a, **k: None)
    sys.modules["runpod"] = runpod
    sys.modules.pop("handler", None)
    import handler
    return handler, calls


class _Pipe:
    """Decoder stdout: hands out exactly `nframes` frames of `fsize` bytes, then a clean EOF.

    `sleep_s` pushes real monotonic time forward INSIDE the decode loop, which is how the decode
    deadline is now driven. Since #105 the budget is checked before decode too (at the nvenc probe), so
    a zero budget no longer reaches the decode branch at all -- it trips one step earlier, and the test
    below would have been asserting about a step it did not name."""

    def __init__(self, nframes, fsize, sleep_s=0.0):
        self._left = nframes
        self._fsize = fsize
        self._sleep_s = sleep_s
        self.closed = False

    def read(self, n):
        if self._left <= 0:
            return b""
        self._left -= 1
        if self._sleep_s:
            time.sleep(self._sleep_s)
        return b"\x00" * n

    def close(self):
        self.closed = True


class _Enc:
    def __init__(self):
        self.stdin = types.SimpleNamespace(write=lambda b: None, close=lambda: None)
        self.written = 0
        self.killed = False

    # `timeout` and `kill` exist because the shipped code now bounds the wait and can kill the child
    # (#105). A fake that does not accept them would make the guard untestable through this harness.
    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


def _wire(handler, monkeypatch, *, nframes, batch_tiles, w=8, h=8, batch_sleep_s=0.0, read_sleep_s=0.0):
    """Point _upscale_video at fakes. `batch_tiles` is the tile each successive batch will SETTLE on,
    so a test can make the first batch shrink and then assert what the second batch was HANDED.

    `batch_sleep_s` may be a float (every batch) or a sequence (per batch, last value sticks).
    The projection tests need the first batch slower than the rest -- that is the OOM-search
    shape -- and a single sleep cannot tell a working projection from one that used the
    search-inflated rate and refused a job that would have finished.
    """
    seen_start_tiles = []
    settled = list(batch_tiles)
    sleeps = list(batch_sleep_s) if isinstance(batch_sleep_s, (list, tuple)) else None

    def fake_upscale_batch(model, frames_np, out_w, out_h, start_tile=None):
        seen_start_tiles.append(start_tile)
        idx = len(seen_start_tiles) - 1
        sleep = sleeps[min(idx, len(sleeps) - 1)] if sleeps is not None else batch_sleep_s
        if sleep:
            time.sleep(sleep)  # push real monotonic time past the deadline inside the LOOP
        tile = settled[min(idx, len(settled) - 1)]
        return [b"o" for _ in frames_np], tile

    def fake_popen(cmd, **kw):
        if "rawvideo" in cmd and cmd[-1] == "-":  # decode: ffmpeg ... -f rawvideo ... -
            return types.SimpleNamespace(stdout=_Pipe(nframes, w * h * 3, read_sleep_s),
                                        wait=lambda timeout=None: 0, kill=lambda: None)
        return _Enc()

    # **kw because the shipped probes now take the invocation deadline (#105); a fake with the old
    # signature would fail the call rather than exercise the loop.
    monkeypatch.setattr(handler, "_ffprobe",
                        lambda src, spec, **kw: (["24/1"] if "frame_rate" in spec else [str(w), str(h)]))
    monkeypatch.setattr(handler, "_nvenc_available", lambda: False)
    monkeypatch.setattr(handler, "_has_audio", lambda src, **kw: False)
    monkeypatch.setattr(handler, "_upscale_batch", fake_upscale_batch)
    monkeypatch.setattr(handler.subprocess, "Popen", fake_popen)
    return seen_start_tiles


def test_control_the_harness_actually_drives_the_real_loop(monkeypatch):
    """CONTROL, first. If this fails, every assertion below is about a loop that never ran."""
    handler, calls = _load_handler({"UPSCALE_BATCH": 2, "UPSCALE_TILE": 512, "FFMPEG_TIMEOUT": 9999})
    seen = _wire(handler, monkeypatch, nframes=4, batch_tiles=[512, 512])
    info = handler._upscale_video(object(), "in.mp4", "out.mp4", 2)
    assert info["frames"] == 4, info
    assert len(seen) == 2, f"expected 2 batches of 2 frames, saw {len(seen)}"
    assert calls["empty_cache"] == 1  # the success path always released, and still does


def test_settled_tile_is_carried_into_the_next_batch(monkeypatch):
    """The first batch shrinks 512 -> 256. The second must be HANDED 256, not sent back to 512."""
    handler, _ = _load_handler({"UPSCALE_BATCH": 2, "UPSCALE_TILE": 512, "FFMPEG_TIMEOUT": 9999})
    seen = _wire(handler, monkeypatch, nframes=6, batch_tiles=[256, 256, 256])
    info = handler._upscale_video(object(), "in.mp4", "out.mp4", 2)
    # Batch 1 gets None (no prior knowledge -> TILE). Batches 2 and 3 get the settled 256.
    assert seen == [None, 256, 256], seen
    assert info["tile_min"] == 256 and info["tile_shrank"] is True


def test_a_shrink_is_never_widened_back_up(monkeypatch):
    """A later batch settling LOWER must not be undone, and the carry must never exceed TILE."""
    handler, _ = _load_handler({"UPSCALE_BATCH": 2, "UPSCALE_TILE": 512, "FFMPEG_TIMEOUT": 9999})
    seen = _wire(handler, monkeypatch, nframes=6, batch_tiles=[256, 128, 128])
    info = handler._upscale_video(object(), "in.mp4", "out.mp4", 2)
    assert seen == [None, 256, 128], seen
    assert info["tile_min"] == 128


def test_the_timeout_path_releases_the_cuda_cache(monkeypatch):
    """THE HEADLINE. Pre-fix `empty_cache` sat after the loop and the raise jumped over it, so this
    count was 0 and ~19.7 GiB stayed reserved for the life of the process.

    The deadline must be tripped INSIDE THE UPSCALE LOOP, which is the path that has something to
    release. The obvious way to force a timeout -- FFMPEG_TIMEOUT=0 -- trips the DECODE deadline
    first and raises `decode exceeded FFMPEG_TIMEOUT`, so this test would have passed while
    exercising an entirely different branch. It was caught only because the assertion names the
    specific refusal instead of accepting any TimeoutError."""
    handler, calls = _load_handler({"UPSCALE_BATCH": 2, "UPSCALE_TILE": 512, "FFMPEG_TIMEOUT": 1})
    _wire(handler, monkeypatch, nframes=6, batch_tiles=[512, 512, 512], batch_sleep_s=1.2)
    # InvocationExpired rather than TimeoutError since #105: TimeoutError is an Exception, so every
    # broad `except Exception` on the job paths swallowed it and re-keyed the degrade.
    with pytest.raises(handler.InvocationExpired) as e:
        handler._upscale_video(object(), "in.mp4", "out.mp4", 2)
    assert "upscale exceeded" in str(e.value), "wrong branch: this is not the leak path"
    assert calls["empty_cache"] == 1, "the abort path must release the allocator's reservation"
    assert calls["synchronize"] == 1


def test_the_decode_timeout_path_is_documented_not_assumed(monkeypatch):
    """The decode deadline exits BEFORE the upscale try/finally, so it does not release -- and that is
    correct rather than an oversight: at that point the only CUDA allocation is the model weights,
    which are ALLOCATED, not cached, and `empty_cache()` does not free them. Asserting the behaviour
    here means the next reader does not have to guess whether this branch was considered."""
    handler, calls = _load_handler({"UPSCALE_BATCH": 2, "UPSCALE_TILE": 512, "FFMPEG_TIMEOUT": 1})
    _wire(handler, monkeypatch, nframes=6, batch_tiles=[512, 512, 512], read_sleep_s=0.6)
    with pytest.raises(handler.InvocationExpired) as e:
        handler._upscale_video(object(), "in.mp4", "out.mp4", 2)
    assert "decode exceeded" in str(e.value)
    assert calls["empty_cache"] == 0


def test_the_release_is_not_conditional_on_which_exit_was_taken(monkeypatch):
    """A non-timeout failure inside the loop must release too. Without this the assertion above is
    satisfied by a fix that special-cases TimeoutError, which would leak on every other error."""
    handler, calls = _load_handler({"UPSCALE_BATCH": 2, "UPSCALE_TILE": 512, "FFMPEG_TIMEOUT": 9999})
    _wire(handler, monkeypatch, nframes=4, batch_tiles=[512, 512])

    def boom(*a, **kw):
        raise RuntimeError("CUDA error: something else entirely")

    monkeypatch.setattr(handler, "_upscale_batch", boom)
    with pytest.raises(RuntimeError, match="something else entirely"):
        handler._upscale_video(object(), "in.mp4", "out.mp4", 2)
    assert calls["empty_cache"] == 1


def test_the_carried_tile_is_clamped_into_the_configured_range():
    """The clamp exists so a carried value can never widen the search past the configured ceiling.

    It is tested at all because the mutation pass showed that DELETING it reddened nothing: it was
    inline inside `_upscale_batch`, which needs torch and numpy to run, so no hermetic test could
    reach it. Lifting it to a pure function is the fix for that, not a tidy-up."""
    handler, _ = _load_handler({"UPSCALE_TILE": 512, "UPSCALE_TILE_FLOOR": 64})
    assert handler._clamped_start_tile(None) == 512   # no prior knowledge -> the configured tile
    assert handler._clamped_start_tile(256) == 256    # the ordinary carry
    assert handler._clamped_start_tile(4096) == 512   # never wider than the ceiling
    assert handler._clamped_start_tile(1) == 64       # never below the floor
    assert handler._clamped_start_tile(64) == 64      # the floor itself is admissible


# ---- the remaining half of #98: project and refuse, never ok:true at the source res ---------------

def test_project_does_not_fire_when_the_work_fits():
    """CONTROL for the method. Without this, a project() that always raises still greens every
    'it refused' test below."""
    handler, _ = _load_handler()
    handler._Deadline(1200).project("upscale", 1.0, 16)


def test_the_projection_reason_fits_the_120_characters_that_survive():
    """Same 120-char cap as check() (vivijure-cf degradeReason). A projected refuse that loses
    'projected' to truncation is indistinguishable from a regular timeout."""
    handler, _ = _load_handler()
    with pytest.raises(handler.InvocationExpired) as e:
        handler._Deadline(1200).project("upscale", 5000, 1440)
    assert len(str(e.value)) <= 120, str(e.value)
    assert "projected" in str(e.value), str(e.value)
    assert str(e.value).startswith("upscale-invocation-guard: upscale projected "), str(e.value)


def test_a_job_that_cannot_finish_is_refused_after_the_settled_batch_not_at_the_budget(monkeypatch):
    """THE HEADLINE. Ten batches, each 0.35s, budget 3s. A regular deadline.check fires at 3s
    after ~8 batches. A working projection fires after batch 1 (tile already settled at TILE,
    so that batch IS the rate) and before batch 2 burns the rest of the card.

    Elapsed MUST be asserted. InvocationExpired arrives either way -- 3s later if project()
    is deleted -- and that is the same trap #105's encode test exists for.
    """
    handler, calls = _load_handler({"UPSCALE_BATCH": 2, "UPSCALE_TILE": 512})
    _wire(handler, monkeypatch, nframes=20, batch_tiles=[512], batch_sleep_s=0.35)
    t0 = time.monotonic()
    with pytest.raises(handler.InvocationExpired) as e:
        handler._upscale_video(object(), "in.mp4", "out.mp4", 2, budget=3)
    elapsed = time.monotonic() - t0
    assert "projected" in str(e.value), f"wrong branch (regular timeout?): {e.value}"
    assert elapsed < 1.2, f"burned the budget instead of projecting: {elapsed:.2f}s"
    assert elapsed >= 0.30, f"refused before a settled batch existed: {elapsed:.2f}s"
    assert calls["empty_cache"] == 1, "projection is an abort path; it must release too"


def test_a_first_batch_that_shrank_must_not_project_the_search_rate(monkeypatch):
    """x4plus at 720p settles 512 -> 256 on batch 1. That batch's seconds-per-frame includes
    a forward pass known to OOM. Projecting from it refuses shots the remaining batches would
    have finished (the 7s films in our own render history).

    First batch 1.4s (search), then 0.08s. Budget 3s. Search-rate projection after batch 1:
    1.4/2 = 0.7s/frame * 8 left = 5.6s > ~1.6s remaining -> refuse. Settled-rate projection
    after batch 2: 0.08/2 = 0.04s/frame * 6 left = 0.24s < remaining -> admit. Completes.
    """
    handler, _ = _load_handler({"UPSCALE_BATCH": 2, "UPSCALE_TILE": 512})
    seen = _wire(handler, monkeypatch, nframes=10, batch_tiles=[256, 256, 256, 256, 256],
                 batch_sleep_s=[1.4, 0.08])
    t0 = time.monotonic()
    info = handler._upscale_video(object(), "in.mp4", "out.mp4", 2, budget=3)
    elapsed = time.monotonic() - t0
    assert info["frames"] == 10, info
    assert seen == [None, 256, 256, 256, 256], seen
    assert elapsed < 2.4, f"should have completed on the settled rate: {elapsed:.2f}s"


def test_a_job_that_cannot_finish_returns_ok_false_and_never_uploads(monkeypatch):
    """The job-path shape. A long shot must not come back as ok:true (with or without a
    scale / out_h that claims the upscale happened). detail, not error: error is lifted
    into a FAILED envelope and books a failed job row for an honest degrade.
    """
    handler, _ = _load_handler({
        "UPSCALE_BATCH": 2, "UPSCALE_TILE": 512,
        "R2_ENDPOINT_URL": "https://r2.invalid",
        "R2_ACCESS_KEY_ID": "test",
        "R2_SECRET_ACCESS_KEY": "test",
    })
    _wire(handler, monkeypatch, nframes=20, batch_tiles=[512], batch_sleep_s=0.35)
    monkeypatch.setattr(handler, "_invocation_budget", lambda: 3)
    uploaded = []
    monkeypatch.setattr(handler, "_r2", lambda: types.SimpleNamespace(
        download_file=lambda *a, **k: None,
        upload_file=lambda *a, **k: uploaded.append(True),
        put_object=lambda *a, **k: None,
    ))
    monkeypatch.setattr(handler, "_load_model", lambda name: object())
    out = handler._upscale_r2({
        "project": "p",
        "clip_key": "renders/p/clips/s.mp4",
        "output_key": "renders/p/clips/s_up.mp4",
        "model": "RealESRGAN_x4plus",
        "scale": 2,
    })
    assert out["ok"] is False, out
    assert "error" not in out, out
    assert "projected" in out.get("detail", ""), out
    assert len(out["detail"]) <= 120
    assert uploaded == [], "must not upload a partial or source-resolution artifact"
    assert "scale" not in out and "out_h" not in out, out
    assert "applied" not in out, out
