"""Unit coverage for the homelab serve route (`runpod_http_serve.route`), which had NONE.

Measured before writing this: `runpod_http_serve` appeared in `Dockerfile.serve` and `serve.py`
and in ZERO test files, in this repo and in `vivijure-audio-upscale` both. The serve layer is what
stands between an open GPU endpoint and the VLAN, and it was entirely untested.

The point of the first test is #88: `POST /run {"selftest": true}` must be FORWARDED to the wrapped
handler, never answered here. `handler._selftest` is the documented deploy-verification GPU check;
an interception at this layer returns ok:true on a box with no GPU, a broken model, a missing
weight or a dead ffmpeg -- a deploy check structurally incapable of failing.

No stubs are needed: `runpod_http_serve` is stdlib-only by design, so this exercises the SHIPPED
module rather than a stand-in.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import runpod_http_serve as rhs  # noqa: E402

TOKEN = "test-token-not-a-real-credential"
SERVICE = "vivijure-upscale-finish-upscale"


def _recording_registry():
    """A registry whose run_fn RECORDS what the handler layer was actually handed."""
    seen = []

    def run_fn(payload, _should_cancel):
        seen.append(payload)
        return {"ok": True, "echo": payload}

    return rhs.JobRegistry(run_fn), seen


def _drain(registry, job_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = registry.get(job_id)
        if job is not None and job.status in (rhs.JobStatus.COMPLETED, rhs.JobStatus.FAILED):
            return job
        time.sleep(0.02)
    return registry.get(job_id)


def _post_run(registry, body, token=TOKEN):
    return rhs.route("POST", "/run", body, registry=registry, token=token,
                     expected_token=TOKEN, service=SERVICE)


# ---- #88: selftest must reach the handler ---------------------------------------------------------

def test_selftest_is_forwarded_to_the_handler_not_intercepted():
    reg, seen = _recording_registry()
    status, body = _post_run(reg, {"input": {"selftest": True}})

    assert status == 200
    # A JOB was created. The removed intercept returned {"ok": True, "selftest": True, "service": ...}
    # with NO id, so this is the assertion that goes red if it ever comes back.
    assert "id" in body, f"selftest was answered at the route layer instead of submitted: {body}"
    assert "ok" not in body

    job = _drain(reg, body["id"])
    assert job is not None and job.status is rhs.JobStatus.COMPLETED
    # ...and the handler layer really received the selftest payload, which is the NAMED reason this
    # test exists. "an id came back" alone would pass against a registry that dropped the payload.
    assert seen == [{"selftest": True}], seen


def test_selftest_at_the_top_level_is_also_forwarded():
    """The removed intercept checked BOTH `body["selftest"]` and `payload["selftest"]`, so a
    top-level spelling has to be covered too or half the intercept could return unnoticed."""
    reg, seen = _recording_registry()
    status, body = _post_run(reg, {"selftest": True})
    assert status == 200 and "id" in body
    job = _drain(reg, body["id"])
    assert job is not None and job.status is rhs.JobStatus.COMPLETED
    assert seen == [{"selftest": True}], seen


def test_an_ordinary_job_is_forwarded_unchanged():
    """Control: forwarding is not special-cased for selftest. If this failed while the two above
    passed, the route would be doing something selftest-specific rather than nothing at all."""
    reg, seen = _recording_registry()
    payload = {"project": "p", "clip_key": "renders/p/clips/s.mp4", "scale": 2}
    status, body = _post_run(reg, {"input": payload})
    assert status == 200 and "id" in body
    _drain(reg, body["id"])
    assert seen == [payload], seen


# ---- auth ------------------------------------------------------------------------------------------

def test_health_is_auth_free_and_touches_nothing():
    reg, seen = _recording_registry()
    status, body = rhs.route("GET", "/health", None, registry=reg, token=None,
                             expected_token=TOKEN, service=SERVICE)
    assert status == 200 and body["ok"] is True and body["service"] == SERVICE
    # /health must NOT be evidence about the card: nothing reached the handler.
    assert seen == []


def test_run_refuses_a_wrong_token():
    reg, seen = _recording_registry()
    status, body = _post_run(reg, {"input": {"selftest": True}}, token="wrong-token")
    assert status == 401 and body["ok"] is False
    assert seen == []


def test_run_refuses_a_missing_token():
    reg, seen = _recording_registry()
    status, body = _post_run(reg, {"input": {"selftest": True}}, token=None)
    assert status == 401 and body["ok"] is False
    assert seen == []


def test_run_refuses_when_no_token_is_configured_at_all():
    """An unset LOCAL_FINISH_TOKEN must refuse, not fall open. Note this returns 503, NOT 401: the
    two are different findings and collapsing them would hide a misconfigured door."""
    reg, _ = _recording_registry()
    status, body = rhs.route("POST", "/run", {"input": {}}, registry=reg, token="anything",
                             expected_token="", service=SERVICE)
    assert status == 503 and body["ok"] is False


def test_status_of_an_unknown_job_is_404():
    """This is what makes cross-door job-id affinity measurable: a door that did not run a job
    must say so, rather than answering for a sibling."""
    reg, _ = _recording_registry()
    status, _body = rhs.route("GET", "/status/deadbeefdeadbeef", None, registry=reg,
                              token=TOKEN, expected_token=TOKEN, service=SERVICE)
    assert status == 404
