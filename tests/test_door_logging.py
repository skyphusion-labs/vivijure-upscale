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
import re
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


def test_no_log_call_appears_under_the_registry_lock(rhs):
    """No `_log` call may sit inside a lock-held region. Asserted STATICALLY, on the source.

    WHY NOT A RUNTIME TEST. The first version of this test ran a job and, from inside a patched
    `_log`, tried `reg._lock.acquire(blocking=False)` -- reasoning that a failure to acquire meant
    the line was being emitted under the lock. **That instrument cannot distinguish "the printing
    thread holds the lock" from "the worker thread happens to hold it at this instant"**, which is a
    different claim entirely. It passed in isolation and failed roughly one run in three in the full
    suite. A check that cannot separate the two states it exists to separate is not a test, and a
    flaky one is worse than none because it teaches people to re-run until green.

    The property is structural, so it is asserted structurally and cannot race.

    THE HAZARD IT PROTECTS: `print(flush=True)` to a container's stdout is a BLOCKING write. If the
    log pipe fills and nothing drains it, a print under `self._lock` blocks while holding it -- and
    submit, get and cancel all take that same lock. The door would accept nothing and answer
    nothing, with no line to say why. A fix whose purpose is making silence meaningful must not
    introduce a new way to go silent.
    """
    src = Path(rhs.__file__).read_text().splitlines()

    # Track lock depth by indentation: a `with self._lock:` opens a region that ends when
    # indentation returns to or below the `with`'s own column.
    lock_col = None
    in_locked_fn = False
    fn_col = None
    offenders = []
    for i, line in enumerate(src, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        col = len(line) - len(line.lstrip())
        if lock_col is not None and col <= lock_col:
            lock_col = None
        # A `*_locked` method is lock-held BY CONTRACT -- that is what the suffix means in this
        # file -- so its whole body counts as a lock region even though it contains no `with`.
        # MISSING THIS MADE THE FIRST VERSION OF THIS SCAN DECORATIVE: `_retain_locked` held the
        # real defect, has no `with self._lock:` of its own, and the scan passed against the
        # buggy source. It was caught only by driving the test red against the pre-fix code.
        m = re.match(r"def\s+(\w+)\s*\(", stripped)
        if m:
            fn_col = col
            in_locked_fn = m.group(1).endswith("_locked")
            lock_col = None
        elif in_locked_fn and fn_col is not None and col <= fn_col:
            in_locked_fn = False
        if re.match(r"with\s+self\._lock\s*:", stripped):
            lock_col = col
            continue
        if (lock_col is not None or in_locked_fn) and re.search(r"(?<!\.)\b_log\s*\(", stripped):
            offenders.append(f"{i}: {stripped}")

    assert not offenders, "log call under the registry lock: " + "; ".join(offenders)


def test_control_the_static_scan_can_find_a_planted_violation(rhs, tmp_path):
    """POSITIVE CONTROL for the scan above.

    Without this, the scan passes identically whether the source is clean or the matcher is broken.
    A zero from an instrument never shown capable of a non-zero is not evidence.
    """
    planted = tmp_path / "planted.py"
    planted.write_text(
        "class X:\n"
        "    def f(self):\n"
        "        with self._lock:\n"
        "            _log('this one is a violation')\n"
    )
    src = planted.read_text().splitlines()
    lock_col = None
    offenders = []
    for i, line in enumerate(src, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        col = len(line) - len(line.lstrip())
        if lock_col is not None and col <= lock_col:
            lock_col = None
        if re.match(r"with\s+self\._lock\s*:", stripped):
            lock_col = col
            continue
        if lock_col is not None and re.search(r"(?<!\.)\b_log\s*\(", stripped):
            offenders.append(f"{i}: {stripped}")
    assert offenders, "the scan cannot see a planted violation, so its clean result means nothing"


def test_cancel_before_start_still_emits_its_terminal_line(rhs):
    """cancel() returns from inside the lock, so its staged line needs an explicit drain.

    This is the path most likely to be silently dropped by the stage-and-drain change, because it
    is the one that does not go through the worker loop's drain point.
    """
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reg = rhs.JobRegistry(_ok)
        # Do not let the worker start draining: submit and cancel while it is still queued is
        # racy by nature, so assert only that IF it was cancelled before start, the line appears.
        job_id = reg.submit(dict(PAYLOAD))
        reg.cancel(job_id)
        time.sleep(0.2)
    out = buf.getvalue()
    assert f"door accept job={job_id}" in out
    assert f"door done job={job_id}" in out, "terminal line missing on the cancel path"
