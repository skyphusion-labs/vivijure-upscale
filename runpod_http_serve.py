"""RunPod-compatible job API over stdlib HTTP for homelab LOCAL_FINISH_* services.

Sidecars POST /run {"input": {...}} -> {"id"} and poll GET /status/<id> with the same envelope
RunPod serverless uses (IN_QUEUE / IN_PROGRESS / COMPLETED / FAILED). Stdlib only.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import signal
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable


class JobStatus(str, Enum):
    IN_QUEUE = "IN_QUEUE"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Cancelled(Exception):
    pass


def _log(msg: str) -> None:
    """One-line stdout log. flush=True is mandatory: stdout is block-buffered when it is not a TTY,
    which is always the container case, so an unflushed line can sit in the buffer for the whole
    job and the log is useless exactly when it is needed."""
    print(f"door {msg}", flush=True)


@dataclass
class Job:
    id: str
    payload: dict
    status: JobStatus = JobStatus.IN_QUEUE
    output: dict | None = None
    error: str | None = None
    _cancel: bool = field(default=False, repr=False)
    # Observability only (cf#507). queued_at is set at construction, started_at when the worker
    # picks the job up, so elapsed can distinguish "sat in the queue" from "ran a long time".
    queued_at: float = field(default_factory=time.monotonic, repr=False)
    started_at: float | None = field(default=None, repr=False)

    def status_dict(self) -> dict:
        d: dict = {"id": self.id, "status": self.status.value}
        if self.status is JobStatus.COMPLETED and self.output is not None:
            d["output"] = self.output
        if self.status is JobStatus.FAILED and self.error is not None:
            d["error"] = self.error
        return d


RunFn = Callable[[dict, Callable[[], bool]], dict]


class JobRegistry:
    def __init__(self, run_fn: RunFn, *, max_completed: int = 256) -> None:
        self._run_fn = run_fn
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._queue: deque[str] = deque()
        self._completed_order: deque[str] = deque()
        self._max_completed = max_completed
        self._worker: threading.Thread | None = None
        # Terminal log lines staged under the lock and emitted by _drain_logs with it released.
        self._pending_logs: list[str] = []
        self._wake = threading.Condition(self._lock)
        self._stop = False

    def submit(self, payload: dict) -> str:
        job = Job(id=uuid.uuid4().hex, payload=payload)
        with self._lock:
            self._jobs[job.id] = job
            self._queue.append(job.id)
            depth = len(self._queue)
            self._ensure_worker_locked()
            self._wake.notify()
        # ACCEPT line. Logged here and not only at the terminal transition on purpose: a terminal
        # line alone cannot distinguish "the door never received this request" from "it received it
        # and died before reaching a terminal state", and those two states reading the same is
        # exactly what made a silent door indistinguishable from an unreachable one (cf#507).
        # The accept line is what makes a MISSING terminal line mean something.
        # depth is the queue length AFTER this job: the door drains strictly FIFO on one worker
        # thread, so depth > 1 is the door being saturated rather than slow.
        _log(f"accept job={job.id} depth={depth}")
        return job.id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        # Single exit so the staged line is emitted with the lock released. Every path below
        # returned True before this change and every path still does; only the exit moved.
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                if job.status is JobStatus.IN_QUEUE:
                    try:
                        self._queue.remove(job_id)
                    except ValueError:
                        pass
                    job.status = JobStatus.FAILED
                    job.error = "canceled before start"
                    self._retain_locked(job_id)
                elif job.status is JobStatus.IN_PROGRESS:
                    job._cancel = True
        self._drain_logs()
        return True

    def _drain_logs(self) -> None:
        """Emit staged lines with the lock RELEASED. Swap-under-lock, print outside.

        Idempotent and safe to call anywhere: an empty buffer is a no-op. That matters because it
        makes a MISSED drain call delay a line rather than lose it -- the next drain flushes it --
        which is the failure direction to prefer for an observability path.
        """
        with self._lock:
            staged = self._pending_logs
            self._pending_logs = []
        for line in staged:
            _log(line)

    def _ensure_worker_locked(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run_loop, name="finish-serve-jobs", daemon=True)
            self._worker.start()

    def _run_loop(self) -> None:
        while True:
            # Single drain point for every terminal transition this loop produces. Each of them
            # either `continue`s or falls out of the try/except, and both routes come back here,
            # so one call covers all four BY CONSTRUCTION rather than by four remembered calls.
            # Placed before the lock is taken, so nothing is ever printed while holding it.
            self._drain_logs()
            with self._lock:
                while not self._queue and not self._stop:
                    self._wake.wait()
                if self._stop and not self._queue:
                    return
                job_id = self._queue.popleft()
                job = self._jobs.get(job_id)
                if job is None or job.status is not JobStatus.IN_QUEUE:
                    continue
                if job._cancel:
                    job.status = JobStatus.FAILED
                    job.error = "canceled before start"
                    self._retain_locked(job_id)
                    continue
                job.status = JobStatus.IN_PROGRESS
                job.started_at = time.monotonic()
            try:
                output = self._run_fn(job.payload, lambda: self._is_cancelled(job_id))
                with self._lock:
                    job.output = output
                    job.status = JobStatus.COMPLETED
                    self._retain_locked(job_id)
            except Cancelled:
                with self._lock:
                    job.status = JobStatus.FAILED
                    job.error = "canceled"
                    self._retain_locked(job_id)
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    job.status = JobStatus.FAILED
                    job.error = str(e)[:500]
                    self._retain_locked(job_id)

    def _is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job._cancel)

    def _retain_locked(self, job_id: str) -> None:
        # TERMINAL line. Deliberately here rather than at each of the five call sites: every
        # terminal transition (completed, failed, canceled-before-start, canceled, exception)
        # already funnels through this method, so one line covers all of them BY CONSTRUCTION and
        # a branch added later cannot silently skip it. Five hand-placed prints could, and the
        # branch nobody remembers is the one that goes unobserved.
        # Never logs the payload or any part of it: input carries R2 URLs and a bearer.
        job = self._jobs.get(job_id)
        if job is not None:
            started = job.started_at
            elapsed = time.monotonic() - (started if started is not None else job.queued_at)
            ran = "yes" if started is not None else "no"
            # STAGED, never printed here. This method is lock-held by contract (its name says so),
            # and print(flush=True) to a container stdout is a BLOCKING write: if the log pipe
            # fills and nothing drains it, printing here would block while holding self._lock, and
            # submit/get/cancel all take that same lock. The door would accept nothing and answer
            # nothing, with no line to say why -- a fix for silence must not add a way to go silent.
            self._pending_logs.append(
                f"done job={job_id} status={job.status.value} ran={ran} elapsed={elapsed:.1f}s"
            )
        self._completed_order.append(job_id)
        while len(self._completed_order) > self._max_completed:
            old = self._completed_order.popleft()
            self._jobs.pop(old, None)

    def shutdown(self) -> None:
        with self._lock:
            self._stop = True
            self._wake.notify_all()


_STATUS_RE = re.compile(r"^/status/([A-Za-z0-9]+)$")
_CANCEL_RE = re.compile(r"^/cancel/([A-Za-z0-9]+)$")


def token_error(headers_token: str | None, expected: str) -> tuple[int, dict] | None:
    if not expected:
        return 503, {"ok": False, "error": "LOCAL_FINISH_TOKEN not configured: refusing open GPU endpoint"}
    if not headers_token or not hmac.compare_digest(headers_token, expected):
        return 401, {"ok": False, "error": "unauthorized"}
    return None


MAX_HTTP_BODY_BYTES = 1_048_576  # 1 MiB cap on POST /run (K3: memory DoS)


class _BodyRejected:
    """A request body refused before use, kept DISTINCT from an absent body.

    #106: _body() answered None for three different situations (no body at all, a body
    past the cap, and a body that would not parse), and the /run branch then did
    (body or {}).get("input", body or {}), so all three became an EMPTY job accepted with
    200 and a job id. The caller got a success shape for a request that was never honoured,
    and the job failed later complaining about a missing key rather than about its body.

    The memory-DoS half of the cap always worked, since an oversize body is never read into
    memory (_body returns before rfile.read), and that is exactly why the semantics half
    went unnoticed. Ported from vivijure-blender (fc#1592).
    """

    __slots__ = ("code", "message")

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message


BODY_TOO_LARGE = _BodyRejected(413, "request body exceeds %d bytes" % MAX_HTTP_BODY_BYTES)
BODY_INVALID = _BodyRejected(400, "request body is not valid JSON")


def route(
    method: str,
    path: str,
    body: dict | None | _BodyRejected,
    *,
    registry: JobRegistry,
    token: str | None,
    expected_token: str,
    service: str,
    version: str = "serve-1",
) -> tuple[int, dict]:
    if method == "GET" and path == "/health":
        return 200, {"ok": True, "service": service, "version": version, "mode": "local-finish-http"}

    if method == "POST" and path == "/run":
        err = token_error(token, expected_token)
        if err:
            return err
        # Refuse a rejected body instead of silently running an EMPTY job (#106).
        # AFTER the auth check on purpose: an unauthenticated caller still gets 401 and
        # learns nothing about the cap, preserving the vivijure-musetalk#93 ordering.
        if isinstance(body, _BodyRejected):
            return body.code, {"ok": False, "error": body.message}
        payload = (body or {}).get("input", body or {})
        # #88 / fc#1592: selftest is FORWARDED to the wrapped handler like any other job, never
        # intercepted here. `handler._selftest` is the documented deploy-verification GPU check --
        # it loads every shipped model, generates a real multi-second clip and upscales it on the
        # card. Answering at this layer without reaching the handler made that check STRUCTURALLY
        # INCAPABLE OF FAILING: it returned ok:true on a box with no GPU, a broken model, a missing
        # weight or a dead ffmpeg, which is the one thing a deploy check must never do. Ported from
        # vivijure-audio-upscale, where the identical intercept was found and fixed first.
        # /health remains the fast auth-free liveness probe -- unchanged, and deliberately NOT the
        # thing that proves the card works.
        job_id = registry.submit(payload)
        return 200, {"id": job_id}

    m = _STATUS_RE.match(path)
    if method == "GET" and m:
        err = token_error(token, expected_token)
        if err:
            return err
        job = registry.get(m.group(1))
        if job is None:
            return 404, {"status": 404, "title": "Not Found", "detail": "job not found"}
        return 200, job.status_dict()

    m = _CANCEL_RE.match(path)
    if method == "POST" and m:
        err = token_error(token, expected_token)
        if err:
            return err
        registry.cancel(m.group(1))
        return 200, {"ok": True}

    return 404, {"status": 404, "title": "Not Found", "detail": "no such route"}


def wrap_runpod_handler(handler_fn: Callable[[dict], dict]) -> RunFn:
    """Adapt a RunPod handler(job) to the registry run_fn(payload, should_cancel)."""

    def run(payload: dict, should_cancel: Callable[[], bool]) -> dict:
        if should_cancel():
            raise Cancelled()
        job = {"input": payload}
        result = handler_fn(job)
        if not isinstance(result, dict):
            raise RuntimeError(f"handler returned non-dict: {type(result).__name__}")
        # RunPod marks top-level `error` as FAILED; soft-degrade uses `detail` only.
        if result.get("error"):
            raise RuntimeError(str(result["error"])[:500])
        return result

    return run


def run_serve(
    handler_fn: Callable[[dict], dict],
    *,
    service: str,
    host: str | None = None,
    port: int | None = None,
    token_env: str = "LOCAL_FINISH_TOKEN",
    version: str = "serve-1",
) -> None:
    host = host or os.environ.get("HOST", "0.0.0.0")
    port = int(port or os.environ.get("PORT", "8010") or "8010")
    expected_token = os.environ.get(token_env, "") or ""
    registry = JobRegistry(wrap_runpod_handler(handler_fn))

    class Handler(BaseHTTPRequestHandler):
        def _bearer(self) -> str | None:
            h = self.headers.get("authorization") or ""
            return h[7:] if h.lower().startswith("bearer ") else None

        def _body(self) -> dict | None | _BodyRejected:
            try:
                length = int(self.headers.get("content-length") or 0)
            except (TypeError, ValueError):
                return BODY_INVALID
            if length < 0 or length > MAX_HTTP_BODY_BYTES:
                return BODY_TOO_LARGE
            if not length:
                return None
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                return BODY_INVALID

        def _dispatch(self, method: str) -> None:
            status, payload = route(
                method,
                self.path,
                self._body() if method == "POST" else None,
                registry=registry,
                token=self._bearer(),
                expected_token=expected_token,
                service=service,
                version=version,
            )
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def log_message(self, *args) -> None:
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)

    def _graceful(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _graceful)

    print(f"{service} LOCAL_FINISH HTTP on {host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        httpd.server_close()
        registry.shutdown()
