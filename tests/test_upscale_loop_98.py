"""upscale#98 -- the abort path leaked the CUDA cache, and the tile search was re-paid per batch.

Both defects live in `_upscale_video`'s batch loop, so both are driven THROUGH THAT LOOP here rather
than through a re-statement of it. Heavy deps are stubbed exactly as the sibling tests do (no GPU, no
ffmpeg), but the control flow under test is the shipped one.

The two findings and why they are one issue:

1. `torch.cuda.empty_cache()` sat AFTER the loop with no `finally`, so `raise TimeoutError` jumped
   over it and the allocator kept its reservation for the life of the process. MEASURED on fatmike:
   ~19.7 GiB of a 20475 MiB card still held while the container sat idle with the job long finished,
   against ~2 GiB after a job that COMPLETED. The leak is a property of the ABORT path, not of torch
   and not of the model, so the two halves of #98 are one defect: the timeout causes the leak.

2. `_upscale_batch` restarted its OOM-shrink search from `TILE` on every batch, and the settled tile
   was captured only for reporting. On x4plus at 720p the tile settles 512 -> 256, so every batch
   re-ran a forward pass already known to OOM, caught it, emptied the cache and retried.
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
    so a test can make the first batch shrink and then assert what the second batch was HANDED."""
    seen_start_tiles = []
    settled = list(batch_tiles)

    def fake_upscale_batch(model, frames_np, out_w, out_h, start_tile=None):
        seen_start_tiles.append(start_tile)
        if batch_sleep_s:
            time.sleep(batch_sleep_s)  # push real monotonic time past the deadline inside the LOOP
        tile = settled[min(len(seen_start_tiles) - 1, len(settled) - 1)]
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
