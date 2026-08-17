"""upscale#105 -- the wall-clock guard over the WHOLE invocation, proved on real subprocesses.

WHAT WAS WRONG. `_upscale_video` stamped one deadline and checked it twice (decode, upscale). The
ENCODE step -- a raw `subprocess.Popen`, a write loop, and an unbounded `enc.wait()` -- sat outside it,
so the invocation was not capped by anything. It read as complete because the module carried a helper
named `_run(cmd, timeout=FFMPEG_TIMEOUT)` that looked like the convention the encode path had opted out
of. Measured at d34135d, that helper had ZERO callers in the entire repo: the convention it advertised
was honoured by none of the module's subprocess sites, not one.

WHY THESE TESTS USE REAL SUBPROCESSES. A faked `Popen` proves the decision and skips the mechanism the
guard exists for. The two cases that actually matter -- a write to a pipe nothing is draining, and a
`wait()` on a child that will not exit -- both block in the KERNEL, where no between-steps check is ever
reached. Only killing the child unblocks them. A fake `write` that returns immediately cannot tell a
working guard from a missing one, so every interrupt test below drives a real `sleep` process.

WHY THEY ASSERT ELAPSED TIME, NOT JUST THE OUTCOME. The outcome arrives either way, just later: a slow
child eventually exits on its own and the next deadline check then reports an expired budget, which
looks exactly like the guard working. The exception proves the clock moved, not that the mechanism
fired. Every interrupt test therefore asserts the wall clock, and disarming the watchdog is one of the
mutations these tests are required to catch.
"""
import os
import subprocess
import sys
import time
import types

import pytest

# Captured BEFORE any monkeypatch. `handler.subprocess` IS the stdlib module, so patching
# `handler.subprocess.Popen` replaces it globally, and a fake that then calls `subprocess.Popen` calls
# itself. The real one has to be held from before the patch.
_REAL_POPEN = subprocess.Popen

# Every knob is set EXPLICITLY below, never inherited: sibling test files mutate os.environ at import
# time and do not restore it, so an inherited FFMPEG_TIMEOUT would make these budgets a fact about test
# ordering rather than about the module.
_DEFAULTS = {
    "FFMPEG_TIMEOUT": "1200",
    "UPSCALE_PLATFORM_TIMEOUT": "600",
    "UPSCALE_PLATFORM_MARGIN": "30",
    "UPSCALE_PROBE_TIMEOUT": "60",
    "UPSCALE_BATCH": "16",
    "UPSCALE_TILE": "512",
    "UPSCALE_TILE_FLOOR": "64",
    "R2_ENDPOINT_URL": "https://r2.invalid",
    "R2_ACCESS_KEY_ID": "test",
    "R2_SECRET_ACCESS_KEY": "test",
}


def _load_handler(env=None):
    """Import handler.py with the heavy deps stubbed (no GPU, no torch, no boto3)."""
    for k, v in dict(_DEFAULTS, **{k: str(v) for k, v in (env or {}).items()}).items():
        os.environ[k] = v

    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = "0-stub"
    torch_mod.float16 = "f16"
    torch_mod.inference_mode = lambda *a, **k: (lambda f: f)
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False, empty_cache=lambda: None, synchronize=lambda: None)
    np_mod = types.ModuleType("numpy")
    np_mod.uint8 = "u8"
    np_mod.frombuffer = lambda b, dtype=None: types.SimpleNamespace(reshape=lambda *a: b)
    np_mod.ascontiguousarray = lambda x: types.SimpleNamespace(tobytes=lambda: x)
    boto3_mod = types.ModuleType("boto3")
    boto3_mod.client = lambda *a, **k: None
    spandrel_mod = types.ModuleType("spandrel")
    spandrel_mod.ModelLoader = object
    runpod_mod = types.ModuleType("runpod")
    runpod_mod.serverless = types.SimpleNamespace(start=lambda *a, **k: None)
    for name, mod in (("torch", torch_mod), ("numpy", np_mod), ("boto3", boto3_mod),
                      ("requests", types.ModuleType("requests")), ("spandrel", spandrel_mod),
                      ("runpod", runpod_mod)):
        sys.modules[name] = mod
    sys.modules.pop("handler", None)
    import handler
    return handler


# ---- the arithmetic, as executable assertions -----------------------------------------------------
# Both bounds are about numbers that live in OTHER repositories, which is exactly why they are asserted
# here: nothing else in the estate compares them, and each was wrong at least once.

def test_control_the_module_under_test_is_the_shipped_one():
    """CONTROL, first. If the stubbed import is not the real handler, every assertion below is about
    something else."""
    handler = _load_handler()
    assert handler.__file__.endswith("handler.py")
    assert callable(handler._invocation_budget)
    # NOT an Exception subclass, asserted directly. The job boundaries name InvocationExpired ahead of
    # their broad handler, so ordering alone hides a regression here: making it Exception-derived
    # reddens nothing on those paths and still leaves every handler that does NOT name it -- including
    # _selftest, which has no boundary of its own -- swallowing the expiry.
    assert issubclass(handler.InvocationExpired, BaseException)
    assert not issubclass(handler.InvocationExpired, Exception)


def test_the_default_budget_fires_under_both_readings_of_a_timeout_zero_endpoint():
    """Endpoint 4q8idwbk6tyqbq reports `timeout: 0`, and that value has two live readings: the RunPod
    dashboard's documented 600s default, or no platform limit at all. A guard above 600 is decoration
    under the first reading -- the platform kills the job first, with no structured output, which the
    studio must treat as a crash and which fails the whole film. FFMPEG_TIMEOUT alone defaults to 1200,
    twice the ceiling deploy.sh sets. The budget has to be under 600 to fire under BOTH readings."""
    handler = _load_handler()
    budget = handler._invocation_budget()
    assert budget == 570, budget
    assert budget < 600, "a guard above the platform ceiling can never fire"


def test_three_attempts_at_the_budget_stay_inside_the_core_phase_floor():
    """vivijure-core sizes a phase stall ceiling as max(PHASE_HARD_DEADLINE_SECONDS,
    FINISH_STEP_MAX_ATTEMPTS * longest declared) -- src/film-model.ts:984, measured at c64770f. It
    FLOORS the ceiling; it does not cap it. So a budget large enough that 3 * budget exceeds 5400 does
    not fail a check anywhere: it silently RAISES the stall ceiling for every film whose chain contains
    this door. Keeping under it is about not extending everyone else's phase."""
    handler = _load_handler()
    assert 3 * handler._invocation_budget() < 90 * 60


def test_the_serve_deployment_keeps_the_full_budget_and_still_fits_the_core_floor():
    """UPSCALE_PLATFORM_TIMEOUT=0 means nothing outside will kill the job -- the homelab serve door,
    where Dockerfile.serve pins it. That deployment keeps FFMPEG_TIMEOUT whole, so the RunPod-shaped
    ceiling never silently shortens a door that has no such ceiling."""
    handler = _load_handler({"UPSCALE_PLATFORM_TIMEOUT": 0})
    assert handler._invocation_budget() == 1200
    assert 3 * 1200 < 90 * 60


@pytest.mark.parametrize("ffmpeg_timeout,expected", [(1200, 570), (600, 570), (300, 300), (60, 60)])
def test_the_budget_is_the_smaller_of_the_two_ceilings(ffmpeg_timeout, expected):
    """A non-default probe on both sides of the crossover: below the platform margin the operator's own
    number wins, above it the platform's does. On a single default value the two readings are
    byte-identical and neither is being tested."""
    handler = _load_handler({"FFMPEG_TIMEOUT": ffmpeg_timeout})
    assert handler._invocation_budget() == expected


def test_the_degrade_reason_fits_the_120_characters_that_survive():
    """vivijure-cf truncates the reason at 120 chars (modules/_shared/finish-soft-degrade.ts:77,
    measured at f1b02d7), so the guard name, the step and the elapsed seconds have to be inside the
    first 120 or the operator gets a sentence with the facts cut off the end."""
    handler = _load_handler()
    reason = handler._Deadline(570).reason("encode")
    assert len(reason) <= 120, reason
    assert reason.startswith("upscale-invocation-guard: encode "), reason
    assert "elapsed" in reason and "570s budget" in reason, reason


# ---- the mechanism, on real child processes -------------------------------------------------------

def _sleeper(**kw):
    """A real child that never reads its stdin and never writes its stdout: the shape of an ffmpeg that
    has stopped making progress. 30s is far longer than any budget here, so a test that waits for it to
    exit on its own fails on the wall clock rather than passing late."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], **kw)


def test_a_real_child_blocked_on_a_full_pipe_is_killed_at_the_budget():
    """THE CASE A FAKE CANNOT SHOW. Writing to a pipe nothing drains blocks in the kernel once the
    64KiB buffer fills; no between-frames deadline check is ever reached again. Killing the child is
    what unblocks the write."""
    handler = _load_handler()
    deadline = handler._Deadline(1)
    proc = _sleeper(stdin=subprocess.PIPE)
    t0 = time.monotonic()
    try:
        with handler._guarded_child(proc, deadline, "encode") as guard:
            with pytest.raises((BrokenPipeError, OSError)):
                for _ in range(200):
                    proc.stdin.write(b"\x00" * 65536)
                    proc.stdin.flush()
        elapsed = time.monotonic() - t0
        # ELAPSED FIRST, deliberately. Disarming the watchdog still ends in a BrokenPipeError -- 30
        # seconds later, when the child exits on its own -- so `fired` and the exception both arrive
        # eventually. The wall clock is the only assertion that separates interrupted from merely late.
        assert elapsed < 6, f"the write blocked past the budget: {elapsed:.1f}s"
        assert elapsed >= 0.9, f"returned before the budget, so nothing was actually blocked: {elapsed:.1f}s"
        assert guard["fired"] is True, "the watchdog never fired; the write was not interrupted"
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_reap_bounds_a_child_that_will_not_exit_and_never_raises():
    """`enc.wait()` and `dec.wait()` were unbounded, including inside the `finally` on the ABORT path --
    the guard's own exit route could hang. _reap runs in a finally, so it must bound AND must not raise
    over the exception that got us there."""
    handler = _load_handler()
    deadline = handler._Deadline(0)
    proc = _sleeper()
    t0 = time.monotonic()
    rc = handler._reap(proc, deadline)
    elapsed = time.monotonic() - t0
    assert elapsed < 6, f"_reap waited on a 30s child: {elapsed:.1f}s"
    assert rc is not None, "the child was killed but never reaped"


# ---- end to end through the shipped _upscale_video -------------------------------------------------

def _wire(handler, monkeypatch, *, nframes, encoder_cmd, w=8, h=8, frame_bytes=65536):
    """Point _upscale_video at a faked DECODER (frames have to come from somewhere) and a REAL ENCODER
    child. The encode half is deliberately not faked: it is the half this issue is about."""
    class _Pipe:
        def __init__(self):
            self._left = nframes

        def read(self, n):
            if self._left <= 0:
                return b""
            self._left -= 1
            return b"\x00" * n

        def close(self):
            pass

    spawned = []

    def fake_popen(cmd, **kw):
        if "rawvideo" in cmd and cmd[-1] == "-":  # the decoder
            return types.SimpleNamespace(stdout=_Pipe(), wait=lambda timeout=None: 0, kill=lambda: None)
        p = _REAL_POPEN(encoder_cmd, **kw)  # the REAL encoder child
        spawned.append(p)
        return p

    monkeypatch.setattr(handler, "_ffprobe",
                        lambda src, spec, **kw: (["24/1"] if "frame_rate" in spec else [str(w), str(h)]))
    monkeypatch.setattr(handler, "_nvenc_available", lambda: False)
    monkeypatch.setattr(handler, "_has_audio", lambda src, **kw: False)
    monkeypatch.setattr(handler, "_upscale_batch",
                        lambda model, frames_np, out_w, out_h, start_tile=None:
                        ([b"\x00" * frame_bytes for _ in frames_np], 512))
    monkeypatch.setattr(handler.subprocess, "Popen", fake_popen)
    return spawned


def test_control_the_harness_completes_when_the_encoder_actually_drains(monkeypatch):
    """CONTROL, and it is the one that makes the next test mean something. Same wiring, same real-child
    plumbing, an encoder that DRAINS its stdin: the invocation completes. Without this, a test that
    expects an expiry passes just as well against a harness that can only ever expire."""
    handler = _load_handler({"FFMPEG_TIMEOUT": 60, "UPSCALE_BATCH": 4})
    drain = [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"]
    _wire(handler, monkeypatch, nframes=8, encoder_cmd=drain)
    t0 = time.monotonic()
    info = handler._upscale_video(object(), "in.mp4", "out.mp4", 2)
    elapsed = time.monotonic() - t0
    assert info["frames"] == 8, info
    assert info["budget_s"] == 60, "the run must report the CONFIGURED budget, not the literal default"
    assert elapsed < 20, f"the control took {elapsed:.1f}s; it is not measuring a healthy path"


def test_the_encode_step_is_inside_the_budget_end_to_end(monkeypatch):
    """THE HEADLINE. A real encoder that never drains its stdin, driven through the shipped
    _upscale_video. Before #105 this hung until the platform killed the worker, because the encode step
    had no deadline check and no timeout between t2 and t3."""
    handler = _load_handler({"FFMPEG_TIMEOUT": 2, "UPSCALE_BATCH": 4})
    stall = [sys.executable, "-c", "import time; time.sleep(30)"]
    spawned = _wire(handler, monkeypatch, nframes=8, encoder_cmd=stall)
    t0 = time.monotonic()
    with pytest.raises(handler.InvocationExpired) as e:
        handler._upscale_video(object(), "in.mp4", "out.mp4", 2)
    elapsed = time.monotonic() - t0
    assert "encode exceeded" in str(e.value), str(e.value)
    # THE ASSERTION THAT CATCHES A DISARMED WATCHDOG. Without it, an unbounded encode still ends in an
    # InvocationExpired -- 30 seconds later, when the child exits on its own and the next check sees an
    # expired budget. The exception proves the clock moved; only the elapsed time proves the interrupt.
    assert elapsed < 8, f"the encode ran past its budget: {elapsed:.1f}s"
    assert elapsed >= 1.5, f"expired before the budget: {elapsed:.1f}s"
    for p in spawned:
        assert p.poll() is not None, "the encoder child outlived the invocation"


def test_the_decode_step_is_bounded_by_the_same_budget_on_a_real_stalled_child(monkeypatch):
    """The decode read blocks in the kernel exactly as the encode write does, so the between-frames
    check cannot save it either. Same watchdog, other end of the pipeline."""
    handler = _load_handler({"FFMPEG_TIMEOUT": 2, "UPSCALE_BATCH": 4})
    stall = [sys.executable, "-c", "import time; time.sleep(30)"]

    def fake_popen(cmd, **kw):
        return _REAL_POPEN(stall, **kw)

    monkeypatch.setattr(handler, "_ffprobe", lambda src, spec, **kw: (["24/1"] if "frame_rate" in spec else ["8", "8"]))
    monkeypatch.setattr(handler, "_nvenc_available", lambda: False)
    monkeypatch.setattr(handler, "_has_audio", lambda src, **kw: False)
    monkeypatch.setattr(handler.subprocess, "Popen", fake_popen)
    t0 = time.monotonic()
    with pytest.raises(handler.InvocationExpired) as e:
        handler._upscale_video(object(), "in.mp4", "out.mp4", 2)
    elapsed = time.monotonic() - t0
    assert "decode exceeded" in str(e.value), str(e.value)
    assert elapsed < 8, f"the decode ran past its budget: {elapsed:.1f}s"


# ---- the guard against the broad handlers on the job paths ------------------------------------------

R2_JOB = {"project": "p", "clip_key": "renders/p/clips/s.mp4", "output_key": "renders/p/clips/s_up.mp4"}


def _r2_job_path(handler, monkeypatch, raiser):
    monkeypatch.setattr(handler, "_r2", lambda: types.SimpleNamespace(
        download_file=lambda *a, **k: None, upload_file=lambda *a, **k: None,
        put_object=lambda *a, **k: None))
    monkeypatch.setattr(handler, "_load_model", lambda name: object())
    monkeypatch.setattr(handler, "_upscale_video", raiser)
    return handler._upscale_r2(dict(R2_JOB))


def test_the_guard_is_not_swallowed_by_the_broad_except_on_the_r2_job_path(monkeypatch):
    """Every job path in handler.py ends in `except Exception -> {"ok": false, "error": ...}`. A guard
    raising an ordinary Exception is caught there, re-keyed to the legacy shape, and reported as an
    ordinary failure: present, tested, and inert on the exact path it exists for. InvocationExpired is
    BaseException-derived so no handler has to REMEMBER to re-raise it."""
    handler = _load_handler()

    def raiser(*a, **k):
        raise handler.InvocationExpired("upscale-invocation-guard: encode exceeded 570s budget (elapsed 571.0s)")

    out = _r2_job_path(handler, monkeypatch, raiser)
    assert out["ok"] is False
    assert "error" not in out, "a top-level `error` is lifted into a FAILED envelope and books a failed job row"
    assert out["detail"].startswith("upscale-invocation-guard: encode "), out
    assert len(out["detail"]) <= 120


def test_control_the_same_broad_except_still_swallows_an_ordinary_error(monkeypatch):
    """The CONTROL for the test above. Without it, that assertion is equally satisfied by a broad
    handler that has stopped catching anything at all, which would be a much worse defect."""
    handler = _load_handler()

    def raiser(*a, **k):
        raise RuntimeError("ordinary failure")

    out = _r2_job_path(handler, monkeypatch, raiser)
    assert out["ok"] is False
    assert out["error"] == "ordinary failure", out
    assert "detail" not in out, "an ordinary error must keep failing loud, not wear the degrade shape"


def test_the_guard_survives_a_broad_except_a_future_maintainer_writes():
    """The property, stated directly rather than through a call path, so it keeps holding when the call
    paths change. Raised from INSIDE a real broad handler, with a control proving that handler is alive
    and does swallow an ordinary error."""
    handler = _load_handler()
    swallowed = []
    try:
        raise handler.InvocationExpired("guard")
    except Exception:  # noqa: BLE001 -- deliberately the handler shape this must defeat
        swallowed.append("guard")
    except handler.InvocationExpired:
        pass
    assert swallowed == [], "a broad `except Exception` caught the guard"

    try:
        raise ValueError("ordinary")
    except Exception:  # noqa: BLE001
        swallowed.append("ordinary")
    assert swallowed == ["ordinary"], "the control handler is dead; the assertion above proves nothing"


def test_the_presigned_job_path_returns_the_same_degrade_shape(monkeypatch):
    """The R2 path is production, but the presigned path is the same contract and would otherwise drift
    into the legacy shape unnoticed."""
    handler = _load_handler()

    def raiser(*a, **k):
        raise handler.InvocationExpired("upscale-invocation-guard: decode exceeded 570s budget (elapsed 571.0s)")

    class _Resp:
        def raise_for_status(self):
            return None

        def iter_content(self, n):
            return [b"x"]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    def fake_pin(method, url, **k):
        return _Resp()

    monkeypatch.setattr(handler, "_load_model", lambda name: object())
    monkeypatch.setattr(handler, "_upscale_video", raiser)
    monkeypatch.setattr(handler, "_url_error", lambda *a, **k: None)
    monkeypatch.setattr(handler, "_pinned_https", fake_pin)
    out = handler.handler({"input": {
        "video_url": "https://bucket.example/v",
        "output_url": "https://bucket.example/o",
    }})
    assert out["ok"] is False
    assert "detail" in out
    assert "error" not in out


def test_the_selftest_reports_a_guard_expiry_instead_of_escaping(monkeypatch):
    """_selftest has no broad handler of its own, so a BaseException would escape handler() entirely and
    arrive as a structureless FAILED envelope -- the shape this whole change exists to avoid."""
    handler = _load_handler()

    def raiser(*a, **k):
        raise handler.InvocationExpired("upscale-invocation-guard: encode exceeded 570s budget (elapsed 571.0s)")

    monkeypatch.setattr(handler, "_load_model", lambda name: object())
    monkeypatch.setattr(handler, "_nvenc_available", lambda: False)
    monkeypatch.setattr(handler, "_run_bounded", lambda cmd, timeout, **kw: types.SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(handler, "_ffprobe", lambda src, spec, **kw: ["8", "8"])
    monkeypatch.setattr(handler, "_upscale_video", raiser)
    out = handler._selftest_one("realesr-animevideov3", 2)
    assert out["ok"] is False
    assert out.get("guard_expired") is True, out
    assert "encode exceeded" in out["error"], out


def test_every_subprocess_run_in_the_module_passes_a_timeout():
    """The DENOMINATOR, asserted rather than described. The helper this replaces had zero callers, so
    the file read as bounded while every subprocess.run in it went around the helper. This fails the
    moment a new unbounded call is added, which is the only way the count stays true."""
    import handler as _h  # noqa: F401 -- ensure the module file resolves the same way
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "handler.py")
    with open(path, encoding="utf-8") as f:
        body = f.read()
    # One definition site inside _run_bounded, and nothing else may call subprocess.run directly.
    assert body.count("subprocess.run(") == 1, "a subprocess.run went around _run_bounded"
    assert "def _run_bounded(cmd, timeout, **kw):" in body, "the mandatory-timeout helper is gone"
    # Both Popen sites are the decode and encode children, and both are watchdogged.
    assert body.count("subprocess.Popen(") == 2, body.count("subprocess.Popen(")
    assert body.count("_guarded_child(") == 3, "a Popen child is not under the watchdog"
