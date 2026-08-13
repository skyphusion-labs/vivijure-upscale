"""The door's per-request log lines (cf#507).

WHY THIS TEST EXISTS. For six days a healthy door and an unreachable door produced the identical
`docker logs` output: one startup banner. The serve overlay logged nothing per request, so "zero log
lines" was read as "the door was never called" when it was only ever evidence that the container had
nothing to say. That is the whole cost this file exists to prevent, so these assertions are about
OBSERVABILITY being present, not about behaviour.

The negative controls carry equal weight: a log that leaks the payload would be worse than no log at
all, because the payload holds presigned R2 URLs and a bearer.
"""
import contextlib
import importlib.util
import io
import sys
import time
from pathlib import Path

import pytest

SECRET = "Bearer SUPERSECRET-do-not-log"
R2_URL = "https://r2.example/clip.mp4?X-Amz-Signature=deadbeef"
PAYLOAD = {"input": {"video_url": R2_URL, "auth": SECRET}}


@pytest.fixture(scope="module")
def rhs():
    path = Path(__file__).resolve().parent.parent / "runpod_http_serve.py"
    spec = importlib.util.spec_from_file_location("rhs_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    # dataclass type resolution reads sys.modules; without this the import raises.
    sys.modules["rhs_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _drive(rhs, run_fn):
    """Run one job to a terminal state, capturing stdout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reg = rhs.JobRegistry(run_fn)
        job_id = reg.submit(dict(PAYLOAD))
        for _ in range(250):
            job = reg.get(job_id)
            if job and job.status.value in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.02)
        time.sleep(0.1)  # let the worker thread finish its terminal write
    return job_id, buf.getvalue()


def _ok(payload, is_cancelled):
    return {"ok": True}


def _boom(payload, is_cancelled):
    raise RuntimeError("model exploded")


def test_accept_line_is_emitted(rhs):
    # The ACCEPT line is what makes a MISSING terminal line meaningful: without it, "never received"
    # and "received then died before terminal" are the same observation.
    job_id, out = _drive(rhs, _ok)
    assert f"door accept job={job_id}" in out
    assert "depth=" in out


def test_terminal_line_on_success(rhs):
    job_id, out = _drive(rhs, _ok)
    assert f"door done job={job_id} status=COMPLETED" in out
    assert "ran=yes" in out
    assert "elapsed=" in out


def test_terminal_line_on_failure(rhs):
    # A failing job must be as visible as a succeeding one, or the log is only useful when nothing
    # is wrong. This is the branch that matters during an incident.
    job_id, out = _drive(rhs, _boom)
    assert f"door done job={job_id} status=FAILED" in out


@pytest.mark.parametrize(
    "needle",
    [SECRET, "SUPERSECRET", R2_URL, "X-Amz-Signature", "video_url"],
    ids=["bearer", "bearer-substring", "r2-url", "presigned-signature", "payload-key"],
)
def test_payload_never_reaches_stdout(rhs, needle):
    _, out_ok = _drive(rhs, _ok)
    _, out_fail = _drive(rhs, _boom)
    assert needle not in (out_ok + out_fail)


def test_control_the_harness_can_see_a_leak():
    """POSITIVE CONTROL for the five negative tests above.

    Without this, all five pass identically whether the payload is safe or the capture is broken --
    a zero from an instrument never shown capable of a non-zero is not evidence.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(SECRET, flush=True)
    assert SECRET in buf.getvalue()
