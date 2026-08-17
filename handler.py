"""RunPod serverless handler -- Real-ESRGAN (CUDA) video upscaling for Vivijure's `upscale` module (#191).

Replaces the video2x/Vulkan path (RunPod has no working Vulkan stack -- proven 2026-06-20). Same MODELS
(Real-ESRGAN), run through PyTorch/CUDA via spandrel. The transport contract + the {"selftest": true}
harness are UNCHANGED from the Vulkan attempt -- only the engine swapped.

The pipeline is GPU-bound end to end: frames are streamed through ffmpeg pipes (raw rgb24 in and out --
NO per-frame PNG disk roundtrip), upscaled in BATCHES on the GPU (fp16 via autocast), the final-size
rescale runs on the GPU, and the re-encode uses NVENC (`h264_nvenc`) when the card + ffmpeg support it.
Sizing is one function (`_resolve_output_size`). `target_height` is the studio contract (exact
output height via the existing GPU interpolate after native 4x). `scale` is only 2 or 4 -- a
value that is not either is refused, never collapsed. An unsatisfiable request (beyond 4x, a
downscale, a long-edge cap that would change a requested height) returns ok:false rather than a
wrong-sized ok:true. If NVENC is not usable on
this image, encode HONESTLY falls back to a bounded libx264 (the resolution cap keeps the CPU encode
bounded); the chosen encoder is reported in the result so a fallback is never silent.

Job input (R2 finish-chain mode -- shared bucket):
  {
    "project":    "<project>",                               # required -- scopes every renders/ key
    "clip_key":   "renders/<project>/clips/<shot>.mp4",
    "output_key": "renders/<project>/clips/<shot>_up.mp4",   # optional
    "target_height": 1080,                                   # studio sends this; exact output height
    "scale":      2,                                         # 2 or 4 only; ignored when target_height set
    "model":      "realesr-animevideov3"
  }
Job input (presigned mode):
  {
    "video_url":  "<presigned R2 GET of the source clip>",
    "output_url": "<presigned R2 PUT for the result>",
    "output_key": "renders/<project>/clips/<shot>_up.mp4",   # echoed back
    "target_height": 1080,
    "scale":      2,
    "model":      "realesr-animevideov3"
  }
Returns: { ok, output_key, bytes, scale, out_w, out_h, model, frames, encoder } on success. Two non-ok shapes, and
the difference is deliberate (#105):
  { ok: false, detail: "<reason>" }  the WALL-CLOCK GUARD expired -- an honest soft degrade. `detail`
                                    is the key vivijure-cf's degradeReason reads first, and it does
                                    NOT get lifted into a RunPod FAILED envelope, so an honest degrade
                                    stops booking a `failed` job row that the telemetry then believes.
  { ok: false, error:  "<reason>" }  an ordinary error (bad key, no frames, encode rc). Legacy shape,
                                    also recovered by the panel, but via the FAILED envelope.
Either way the module passes the original clip through -- never a drop.

THE WALL-CLOCK GUARD covers the WHOLE of _upscale_video: probe, nvenc probe, decode, upscale, encode,
and the wait() on both children. It is enforced ON the child processes (a watchdog that kills them),
not only by checks between steps, because a write to a full pipe and a wait() on a stalled child both
block in the kernel where the next check is never reached. Its budget is _invocation_budget(), which
sits UNDER the platform execution ceiling rather than above it -- see that function. After the first
settled GPU batch the loop also PROJECTS remaining work against whatever is left of that budget and
refuses immediately if it cannot finish (#98): burning the rest of the guard then degrading is the
same un-upscaled film, just later.
"""

import os

# PYTORCH_CUDA_ALLOC_CONF is read ONCE by torch at import to configure the CUDA caching allocator, so it
# must be set before `import torch` (spandrel/torch below pull it in). expandable_segments:True lets the
# allocator grow and release segments instead of stranding reserved-but-unallocated VRAM as fragmentation
# -- the ~51 GiB reserved-but-free that filled the card at the x4plus OOM (#30). setdefault so an
# operator-set PYTORCH_CUDA_ALLOC_CONF in the endpoint env always wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import contextlib
import ipaddress
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlparse, urlunparse

import boto3
import numpy as np
import requests
import runpod
import torch
from spandrel import ModelLoader

MODELS_DIR = "/models"
MODEL_FILES = {
    "realesr-animevideov3": "realesr-animevideov3.pth",
    "RealESRGAN_x4plus": "RealESRGAN_x4plus.pth",
}
DOWNLOAD_TIMEOUT = 900
UPLOAD_TIMEOUT = 900
TILE = int(os.environ.get("UPSCALE_TILE", "512") or "512")  # tile size (px); env-tunable -- bounds GPU memory per tile pass.
# 512 genuinely subdivides a 720p frame (1280x720 -> 6 tiles); a value >= the frame size makes tiling a
# no-op, which is how a heavy 4x model (RealESRGAN_x4plus RRDB) hit OOM on a full-frame batch (#584 sib).
# When a single frame will not fit even after the batch split has reached 1 (a card too small for one
# frame at TILE through a heavy 4x model, e.g. RealESRGAN_x4plus on a ~48 GB-class card), _upscale_batch
# HALVES the tile and retries, down to this floor -- bounding the spatial size so the frame still upscales
# (slower, correct) instead of hard-failing (#30). Env-tunable.
TILE_FLOOR = int(os.environ.get("UPSCALE_TILE_FLOOR", "64") or "64")
TILE_PAD = 16   # tile overlap to hide seams
# Frames per GPU batch: the model runs on (N,3,h,w) at once instead of a one-at-a-time Python loop, so
# the per-frame launch/Python overhead is amortized and the GPU stays fed. Tune against VRAM.
BATCH = int(os.environ.get("UPSCALE_BATCH", "16") or "16")
# fp16 inference via autocast (weights stay fp32 -- no model.half() fragility). ~2x throughput, less VRAM.
HALF = (os.environ.get("UPSCALE_FP16", "1") or "1").lower() not in ("0", "false", "no", "")
# Cap the OUTPUT long edge (px). 3840 = 2160p / 4K UHD. The model is 4x native, so a 4x of a 1080p
# source would otherwise be 8K (7680x4320); the cap bounds the encode + the in-memory frame buffers
# regardless of source size. Overridable via env for a deliberately larger render.
MAX_LONG_EDGE = int(os.environ.get("MAX_OUTPUT_LONG_EDGE", "3840") or "3840")
# Models are 4x native. A 2x request is 4x inference GPU-downscaled to 2x. Any other integer
# scale used to be silently collapsed to 2 or 4 (#102); that is a billed lie and is refused.
ALLOWED_SCALES = (2, 4)
NATIVE_SCALE = 4
# Wall-clock guard (s) for ONE invocation of _upscale_video -- decode, upscale AND encode, not per
# phase despite the historical name: it is stamped once and shared. A run that blows past it aborts and
# the job degrades (ok:false + detail -> module passthrough) instead of hanging to the platform kill.
FFMPEG_TIMEOUT = int(os.environ.get("FFMPEG_TIMEOUT", "1200") or "1200")
# What the CONTAINER believes the PLATFORM's own execution ceiling to be, in seconds; 0 = none.
# deploy.sh derives it from the same EXECUTION_TIMEOUT_MS it sets on the endpoint so the two cannot
# drift, and Dockerfile.serve pins it to 0 because the homelab serve door has no platform kill at all.
PLATFORM_TIMEOUT = int(os.environ.get("UPSCALE_PLATFORM_TIMEOUT", "600") or "600")
# How far under the platform ceiling this module's own guard sits, so the guard gets there FIRST and
# the outcome is an honest degrade rather than a structureless platform kill.
PLATFORM_MARGIN = int(os.environ.get("UPSCALE_PLATFORM_MARGIN", "30") or "30")
# Bound for the short probe/utility subprocesses (ffprobe, the test clip generator). Every
# subprocess.run in this file passes a timeout; see _run_bounded.
PROBE_TIMEOUT = int(os.environ.get("UPSCALE_PROBE_TIMEOUT", "60") or "60")
# Explicit transport bounds for the R2 client. botocore's own defaults are a bound, but they are a
# default rather than a declaration, and the invocation ceiling has to be readable from this file.
R2_CONNECT_TIMEOUT = int(os.environ.get("R2_CONNECT_TIMEOUT", "30") or "30")
R2_READ_TIMEOUT = int(os.environ.get("R2_READ_TIMEOUT", "120") or "120")
R2_MAX_ATTEMPTS = int(os.environ.get("R2_MAX_ATTEMPTS", "3") or "3")

GUARD_NAME = "upscale-invocation-guard"  # rides in the degrade reason; first 120 chars are all that survive

_MODELS = {}  # name -> loaded spandrel descriptor (warm-worker cache)
_NVENC = None  # tri-state cache: None = unprobed, True/False = h264_nvenc usable on this worker


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _invocation_budget():
    """The wall-clock seconds ONE _upscale_video invocation gets, and why it is not just FFMPEG_TIMEOUT.

    A guard set ABOVE the platform's own execution ceiling never fires: the platform gets there first,
    and its kill produces a job-level FAILED envelope with NO structured output. vivijure-cf's poll-path
    discriminator correctly refuses to absorb that (a real crash must keep failing loud), so the whole
    film dies -- the outcome the guard exists to prevent. MEASURED: deploy.sh has passed
    executionTimeoutMs=600000 (600s) on every endpoint it creates since 2026-07-01, while FFMPEG_TIMEOUT
    defaults to 1200. The guard was set at twice the ceiling it was meant to sit under, so its default
    was decoration on any endpoint this repo's own script deploys.

    With the defaults that resolves to min(1200, 600-30) = 570s, which fires under BOTH readings of an
    endpoint reporting `timeout: 0` (600s platform default, or no platform limit at all). PLATFORM_TIMEOUT
    0 means there is genuinely no platform ceiling -- the homelab serve door -- and that deployment keeps
    the full FFMPEG_TIMEOUT.

    Both bounds this value has to satisfy are asserted in tests/test_invocation_guard_105.py:
      (a) budget < the platform ceiling, or the guard cannot fire;
      (b) FINISH_STEP_MAX_ATTEMPTS (3) * budget < PHASE_HARD_DEADLINE_SECONDS (5400), because
          vivijure-core FLOORS the phase deadline at max(5400, 3 * longest declared) -- it does not cap
          it -- so a larger budget silently RAISES the stall ceiling for every film whose chain contains
          this door, rather than failing a check.
    """
    if PLATFORM_TIMEOUT <= 0:
        return FFMPEG_TIMEOUT
    return max(0, min(FFMPEG_TIMEOUT, PLATFORM_TIMEOUT - PLATFORM_MARGIN))


class InvocationExpired(BaseException):
    """The invocation's wall-clock budget ran out.

    DERIVED FROM BaseException RATHER THAN Exception, and that is the whole point (#105). Every job path
    in this file ends in a broad `except Exception` that turns anything into {"ok": false, "error": ...}.
    A guard whose expiry is an ordinary Exception is swallowed there, re-keyed to the legacy shape, and
    reported as an ordinary failure -- present, tested, and inert on the exact path it was written for.
    BaseException means no existing or future broad handler has to REMEMBER to re-raise it. The three job
    boundaries catch it explicitly and return the soft-degrade shape.

    `finally` blocks still run, so the abort-path CUDA cache release (#98) is unaffected; that is
    asserted rather than assumed.
    """


class _Deadline:
    """One wall-clock budget for the WHOLE invocation: checked between steps, and enforced ON the child
    processes by _guarded_child."""

    def __init__(self, seconds, label=GUARD_NAME):
        self.seconds = int(seconds)
        self.label = label
        self.started = time.monotonic()
        self.expires = self.started + seconds

    def elapsed(self):
        return time.monotonic() - self.started

    def remaining(self):
        return self.expires - time.monotonic()

    def expired(self):
        return time.monotonic() > self.expires

    def child_timeout(self, floor=0.05):
        """Seconds a child may still have. Floored above zero so an already-expired budget still makes a
        real bounded call rather than a zero-timeout one."""
        return max(floor, self.remaining())

    def reason(self, step):
        """The degrade reason. vivijure-cf truncates it at 120 chars (modules/_shared/
        finish-soft-degrade.ts degradeReason), so the guard name, the step and the elapsed seconds are
        ordered to all fit inside the first 120."""
        return f"{self.label}: {step} exceeded {self.seconds}s budget (elapsed {self.elapsed():.1f}s)"[:120]

    def check(self, step):
        if self.expired():
            raise InvocationExpired(self.reason(step))

    def project(self, step, need, frames_left):
        """Refuse NOW if `need` seconds of remaining work will miss the budget (#98).

        Named apart from check() because check() is about the clock already having
        expired, and this is about the clock being about to. The reason has to say
        `projected`, or a test that only looks for InvocationExpired cannot tell a
        working projection from a regular timeout that arrived later.

        Same envelope as check(): InvocationExpired -> job path returns
        {ok: false, detail: ...}, which the module already treats as degrade.
        Never ok:true at the source resolution -- that is the billed lie.
        """
        left = self.remaining()
        if need > left:
            raise InvocationExpired(
                f"{self.label}: {step} projected {need:.0f}s > {left:.0f}s left "
                f"({int(frames_left)} frames)"[:120]
            )


@contextlib.contextmanager
def _guarded_child(proc, deadline, step):
    """Arm a wall-clock watchdog that KILLS `proc` when the budget runs out; disarm it on exit.

    A deadline check between loop iterations cannot interrupt a write to a full pipe or a wait() on a
    stalled child: both block in the kernel and the next check is never reached. Killing the child is
    what unblocks them (the write raises BrokenPipeError, the wait returns), so the guard has to ACT ON
    the process, not merely observe the clock. Yields a state dict whose `fired` is what the caller
    judges on -- a killed child also looks like an ordinary non-zero exit, so rc alone cannot tell the
    two apart, and reporting a kill as `encode pipe exited rc=-9` would hide the guard from the operator
    it fired for."""
    state = {"fired": False, "step": step}

    def _kill():
        state["fired"] = True
        try:
            proc.kill()
        except Exception:  # noqa: BLE001 -- already gone is the outcome we wanted
            pass

    timer = threading.Timer(deadline.child_timeout(), _kill)
    timer.daemon = True
    timer.start()
    try:
        yield state
    finally:
        timer.cancel()


def _reap(proc, deadline):
    """Bounded wait for a child, then kill it if it will not go. NEVER raises: it runs in `finally`
    blocks and must not mask the exception that got us there. Returns the exit code, or None when it
    could not be established."""
    try:
        return proc.wait(timeout=deadline.child_timeout())
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        try:
            return proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            return None
    except Exception:  # noqa: BLE001 -- a stub or an already-reaped child; the caller judges on `fired`
        return None


def _run_bounded(cmd, timeout, **kw):
    """subprocess.run with a MANDATORY wall-clock bound.

    `timeout` has no default ON PURPOSE. The helper this replaces defaulted it to FFMPEG_TIMEOUT and had
    ZERO callers anywhere in the repo (measured at d34135d) -- every subprocess.run in the file went
    around it. A convention nobody calls still reads as one: the file appeared to bound its subprocesses
    because a bounded helper existed next to them. Requiring the argument is what makes an unbounded call
    impossible to write by omission."""
    return subprocess.run(cmd, timeout=timeout, **kw)


def _bounded_probe(cmd, deadline, timeout, step):
    """Run a short probe/utility subprocess under BOTH its own bound and the invocation budget, and
    convert either overrun into the guard's degrade rather than an ordinary error -- a probe that hangs is
    the same operator-visible event as an encode that hangs."""
    t = min(timeout, deadline.child_timeout()) if deadline is not None else timeout
    try:
        return _run_bounded(cmd, timeout=t, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        if deadline is not None and deadline.expired():
            raise InvocationExpired(deadline.reason(step)) from None
        raise InvocationExpired(f"{GUARD_NAME}: {step} exceeded its {int(t)}s bound"[:120]) from None


def _probe_nvenc():
    """True only if h264_nvenc is BOTH listed AND actually encodes on this GPU. An old ffmpeg NVENC API
    (e.g. an old Ubuntu build) can list the encoder yet fail at runtime on some GPU/driver combos, so a
    real test encode is the only honest check; the chosen encoder is reported. Cached on the warm worker."""
    try:
        enc = _run_bounded(["ffmpeg", "-hide_banner", "-encoders"],
                           timeout=30, capture_output=True, text=True)
        if "h264_nvenc" not in (enc.stdout or ""):
            return False
        test = _run_bounded(
            ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "lavfi",
             "-i", "testsrc=size=320x240:rate=10:duration=1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            timeout=60, capture_output=True, text=True)
        return test.returncode == 0
    except Exception:  # noqa: BLE001 -- any probe failure means "not usable", fall back honestly
        return False


def _nvenc_available():
    global _NVENC
    if _NVENC is None:
        _NVENC = _probe_nvenc()
    return _NVENC


def _capped(w, h, max_edge):
    """Clamp (w,h) so the long edge <= max_edge, preserving aspect, and force even dims (yuv420p)."""
    w, h = int(w), int(h)
    longest = max(w, h)
    if max_edge and longest > max_edge:
        r = max_edge / longest
        w, h = max(2, round(w * r)), max(2, round(h * r))
    return w - (w % 2), h - (h % 2)


def _parse_res(res):
    """Parse a selftest "WxH" resolution string to even (w,h); fall back to 720p on anything malformed or
    out of a sane 16..7680 range. Used only by the selftest harness (never the job path)."""
    try:
        ws, hs = str(res).lower().split("x")
        w, h = int(ws), int(hs)
        if 16 <= w <= 7680 and 16 <= h <= 7680:
            return w - (w % 2), h - (h % 2)
    except Exception:  # noqa: BLE001 -- bad input just falls back to the default
        pass
    return 1280, 720


class SizeRequestError(ValueError):
    """Malformed or unsatisfiable size request. The job path returns ok:false + error so the
    module passthroughs the original clip instead of billing a wrong-sized ok:true (#102)."""


def _as_int(value, name):
    """Strict integer parse. bool is a subclass of int in Python, so True would otherwise
    become 1 and silently collapse; refuse it."""
    if isinstance(value, bool) or value is None:
        raise SizeRequestError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != int(value):
            raise SizeRequestError(f"{name} must be an integer, got {value!r}")
        return int(value)
    s = str(value).strip()
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    raise SizeRequestError(f"{name} must be an integer, got {value!r}")


def _parse_size_knobs(inp):
    """Extract (scale, target_height) from a job input.

    `target_height` is the studio contract (vivijure-local local-finish sends 1080 and no
    scale). `scale`, when sent, must be exactly 2 or 4 -- anything else used to be collapsed
    (`4 if int(...) >= 4 else 2`) and that is the billed lie #102 closes. Default scale=2
    only when NEITHER knob is present, so a height-only request is not also a 2x request.
    """
    raw_th = inp.get("target_height") if inp else None
    raw_scale = inp.get("scale") if inp else None
    th = None
    if raw_th is not None and raw_th != "":
        th = _as_int(raw_th, "target_height")
        if th < 2:
            raise SizeRequestError(f"target_height must be >= 2, got {th}")
    scale = None
    if raw_scale is not None and raw_scale != "":
        scale = _as_int(raw_scale, "scale")
        if scale not in ALLOWED_SCALES:
            raise SizeRequestError(
                f"scale must be 2 or 4, got {scale} (refusing to collapse it)")
    if th is None and scale is None:
        scale = 2
    return scale, th


def _report_scale(out_h, src_h):
    """The delivered height ratio, as an int when it is a whole number, else a short float.
    Never the requested knob: a capped 4x that delivered 2.67x must not report scale:4."""
    ratio = out_h / src_h
    rounded = round(ratio, 4)
    if abs(rounded - round(rounded)) < 1e-9:
        return int(round(rounded))
    return rounded


def _applied_tag(out_h, scale_used, *, from_target_height):
    """Honest applied tag. A height request is tagged as the delivered height; a scale
    request is tagged Nx only when that N was actually delivered."""
    if from_target_height or scale_used not in ALLOWED_SCALES:
        return f"upscale:{int(out_h)}h"
    return f"upscale:{int(scale_used)}x"


def _resolve_output_size(src_w, src_h, scale=None, target_height=None, max_edge=None):
    """Single sizing authority. Returns (out_w, out_h, scale_used).

    `target_height` wins when present: deliver that exact even height (aspect-preserved)
    via the existing GPU interpolate after native 4x. Refuse rather than substitute when
    the request is a downscale, equals the source, needs more than 4x, or would be
    changed by MAX_OUTPUT_LONG_EDGE.

    Without `target_height`, `scale` must be 2 or 4. The long-edge cap may shrink a
    scale-only request; the caller reports the ACTUAL out_w/out_h/scale so a cap is
    never a silent lie.

    If both knobs are present they must agree (target_height == even(src_h * scale)).
    """
    if max_edge is None:
        max_edge = MAX_LONG_EDGE
    src_w, src_h = int(src_w), int(src_h)
    if not (src_w > 0 and src_h > 0):
        raise SizeRequestError("could not probe source dimensions")

    if target_height is not None:
        want_h = int(target_height) - (int(target_height) % 2)
        if want_h < 2:
            raise SizeRequestError(f"target_height {target_height} is too small")
        if want_h < src_h:
            raise SizeRequestError(
                f"target_height {target_height} is smaller than source height {src_h} "
                f"(this door upscales; it does not downscale)")
        if want_h == src_h:
            raise SizeRequestError(
                f"target_height {target_height} equals source height {src_h} "
                f"(nothing to upscale)")
        max_h = (src_h * NATIVE_SCALE) - ((src_h * NATIVE_SCALE) % 2)
        if want_h > max_h:
            raise SizeRequestError(
                f"target_height {target_height} needs more than {NATIVE_SCALE}x "
                f"(source {src_h} -> max {max_h})")
        want_w = int(round(src_w * want_h / src_h))
        want_w = want_w - (want_w % 2)
        capped_w, capped_h = _capped(want_w, want_h, max_edge)
        if (capped_w, capped_h) != (want_w, want_h):
            raise SizeRequestError(
                f"target_height {target_height} ({want_w}x{want_h}) exceeds "
                f"MAX_OUTPUT_LONG_EDGE={max_edge}")
        if scale is not None:
            scale_h = (src_h * scale) - ((src_h * scale) % 2)
            if scale_h != want_h:
                raise SizeRequestError(
                    f"target_height {target_height} and scale {scale} disagree "
                    f"(scale would yield height {scale_h} from source {src_h})")
        return want_w, want_h, _report_scale(want_h, src_h)

    if scale is None:
        scale = 2
    if scale not in ALLOWED_SCALES:
        raise SizeRequestError(
            f"scale must be 2 or 4, got {scale} (refusing to collapse it)")
    out_w, out_h = _capped(src_w * scale, src_h * scale, max_edge)
    return out_w, out_h, _report_scale(out_h, src_h)


def _load_model(name):
    name = name if name in MODEL_FILES else "realesr-animevideov3"
    if name not in _MODELS:
        m = ModelLoader().load_from_file(os.path.join(MODELS_DIR, MODEL_FILES[name]))
        m.to(_device()).eval()  # weights fp32; fp16 is applied per-op via autocast in _upscale_batch
        _MODELS[name] = m
    return _MODELS[name]


def _forward_tile(model, t, use_half):
    """Run the model on one (N,3,h,w) tile and return (N,3,h*scale,w*scale), SPLITTING the batch on a
    CUDA out-of-memory error so a heavy model can never hard-OOM. A native-4x RRDB model (x4plus) on a
    16-frame batch of a near-full-frame tile allocated ~46 GiB in one forward and failed every real job
    (#584 sib); tiling bounds the spatial size, this bounds the batch multiple. On OOM: free the cache
    and recurse on halves, down to a single frame. A lone frame that still cannot fit re-raises (the
    caller (_shrink_on_oom) can retry the whole pass at a smaller tile, down to TILE_FLOOR (#30))."""
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_half):
            return model(t).float()
    except RuntimeError as e:  # torch.cuda.OutOfMemoryError is a RuntimeError subclass; match both
        if "out of memory" not in str(e).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # the failing forward left GiB reserved-but-unallocated (fragmentation); reclaim it before the retry
        n = t.shape[0]
        if n <= 1:
            raise
        mid = n // 2
        a = _forward_tile(model, t[:mid], use_half)
        b = _forward_tile(model, t[mid:], use_half)
        return torch.cat([a, b], dim=0)


def _shrink_on_oom(pass_fn, tile, floor, cleanup=None):
    """Run pass_fn(tile); on a CUDA out-of-memory, HALVE the tile (freeing the cache first) and retry,
    down to `floor`. A non-OOM error, or an OOM already at the floor tile, propagates. The small-card
    fallback (#30): once _forward_tile has split the batch to a single frame and that frame STILL will not
    fit, a smaller tile bounds the spatial size so the frame upscales (slower, correct) rather than
    hard-failing. Pure control flow (no torch), unit-tested hermetically like the batch split."""
    while True:
        try:
            return pass_fn(tile)
        except RuntimeError as e:  # torch.cuda.OutOfMemoryError is a RuntimeError subclass; match both
            if "out of memory" not in str(e).lower() or tile <= floor:
                raise
            if cleanup:
                cleanup()  # release the reserved-but-unallocated cache before retrying at a smaller tile
            tile = max(floor, tile // 2)


def _clamped_start_tile(start_tile):
    """The tile a batch should BEGIN its OOM-shrink search at, clamped into [TILE_FLOOR, TILE].

    `None` means no prior knowledge -> start at the configured TILE. Anything else is the tile a
    previous batch of this same job settled on, and it is clamped so a carried value can never widen
    the search past the configured ceiling or drop below the floor. Pure, so it is testable without
    torch -- which it needed to be: the mutation pass showed that deleting the clamp reddened no
    test at all while it lived inline."""
    if start_tile is None:
        return TILE
    return max(TILE_FLOOR, min(TILE, int(start_tile)))


def _tile_pass(model, t, scale, tile, use_half):
    """One full tiled forward of the batch `t` (N,3,h,w) at a given tile size -> (N,3,h*scale,w*scale).
    Each per-tile forward is itself batch-split-on-OOM (_forward_tile); a single-frame OOM at THIS tile
    propagates so _upscale_batch can retry the whole pass at a smaller tile."""
    n, _, h, w = t.shape
    out = torch.zeros((n, 3, h * scale, w * scale), dtype=torch.float32, device=t.device)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y0, x0 = max(y - TILE_PAD, 0), max(x - TILE_PAD, 0)
            y1, x1 = min(y + tile + TILE_PAD, h), min(x + tile + TILE_PAD, w)
            ot = _forward_tile(model, t[:, :, y0:y1, x0:x1], use_half)  # (N,3,th,tw) -> (N,3,th*s,tw*s); OOM-safe
            cy1, cx1 = min(y + tile, h), min(x + tile, w)
            sy, sx = (y - y0) * scale, (x - x0) * scale
            th, tw = (cy1 - y) * scale, (cx1 - x) * scale
            out[:, :, y * scale:cy1 * scale, x * scale:cx1 * scale] = ot[:, :, sy:sy + th, sx:sx + tw]
    return out


@torch.inference_mode()
def _upscale_batch(model, frames_np, out_w, out_h, start_tile=None):
    """Upscale a BATCH of same-size frames on the GPU, tiled to bound memory, then GPU-resize to
    (out_w,out_h). `frames_np` is a list of (h,w,3) uint8 arrays; returns ((N,out_h,out_w,3) uint8 array,
    tile_used). fp16 via autocast when enabled. Starts at `start_tile` (default TILE) and, on a
    single-frame CUDA OOM, shrinks the tile (halving down to TILE_FLOOR) so a small card still
    finishes (#30). No disk, no per-frame round-trip.

    `start_tile` exists because the shrink search was being re-paid on EVERY batch (#98). The tile a
    batch settles on is a property of this card, this model and this frame size, none of which change
    between batches of one job -- so beginning the next batch back at TILE means a forward pass that is
    KNOWN to OOM, its exception, an `empty_cache()`, and a retry, once per batch. Measured on x4plus at
    720p the tile settles 512 -> 256, so a 30s shot pays that dance 45 times. It is clamped into
    [TILE_FLOOR, TILE] so a caller cannot widen the search past the configured ceiling."""
    cuda = torch.cuda.is_available()
    scale = getattr(model, "scale", 4)
    arr = np.stack(frames_np).astype(np.float32) / 255.0      # (N,h,w,3)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(_device())  # (N,3,h,w)
    use_half = HALF and cuda
    tile0 = _clamped_start_tile(start_tile)
    used = {"tile": tile0}  # the tile the successful pass settled on (records a shrink for the report)
    def _pass(tile):
        used["tile"] = tile
        return _tile_pass(model, t, scale, tile, use_half)
    out = _shrink_on_oom(_pass, tile0, TILE_FLOOR,
                         cleanup=(torch.cuda.empty_cache if cuda else None))
    if out.shape[-1] != out_w or out.shape[-2] != out_h:
        out = torch.nn.functional.interpolate(
            out, size=(out_h, out_w), mode="bicubic", align_corners=False, antialias=True)
    out = out.clamp(0, 1).mul_(255.0).add_(0.5).permute(0, 2, 3, 1).to(torch.uint8)
    return out.cpu().numpy(), used["tile"]  # (N,out_h,out_w,3), tile the pass settled on


def _ffprobe(path, entries, deadline=None, timeout=PROBE_TIMEOUT):
    """`deadline` is threaded in from _upscale_video so the probes count against the SAME budget as the
    work they precede; the selftest's reporting probes pass none and take the plain PROBE_TIMEOUT."""
    p = _bounded_probe(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", entries, "-of", "default=nw=1:nk=1", path],
        deadline, timeout, "probe",
    )
    return [ln for ln in (p.stdout or "").strip().splitlines() if ln]


def _has_audio(path, deadline=None, timeout=PROBE_TIMEOUT):
    p = _bounded_probe(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        deadline, timeout, "probe",
    )
    return bool((p.stdout or "").strip())


def _read_exact(stream, n):
    """Read exactly n bytes from a pipe (it can short-read). Returns None at a clean EOF or on a trailing
    partial frame (valid streams deliver whole frames)."""
    parts, got = [], 0
    while got < n:
        chunk = stream.read(n - got)
        if not chunk:
            return None
        parts.append(chunk)
        got += len(chunk)
    return b"".join(parts)


def _upscale_video(model, src, dst, final_scale, budget=None, target_height=None):
    """Decode -> batched GPU upscale + GPU resize -> re-encode, entirely through ffmpeg rawvideo pipes
    (no PNG disk roundtrip). Audio is copied when present. Returns a dict: frames, encoder, out dims,
    the delivered scale, per-phase seconds (decode/upscale/encode), the budget the run was held to,
    and the batch/fp16 settings actually used.

    `final_scale` is 2, 4, or None (height-only request). `target_height` is the studio contract.
    Size is resolved AFTER the source probe by `_resolve_output_size` -- the one authority.

    ONE deadline covers every step, and it is STAMPED FIRST so the probes are inside it too. Before
    #105 it was stamped after the probes and after the nvenc probe, and it was checked in exactly two
    places (decode and upscale): the encode step -- a raw Popen, a write loop, and an unbounded
    enc.wait() -- sat outside it entirely, so the invocation was never capped by anything. `budget`
    is injectable for tests; production passes none and gets _invocation_budget()."""
    deadline = _Deadline(budget if budget is not None else _invocation_budget())
    fps = (_ffprobe(src, "stream=r_frame_rate", deadline=deadline) or ["24/1"])[0]
    wh = _ffprobe(src, "stream=width,height", deadline=deadline)
    sw, sh = (int(wh[0]), int(wh[1])) if len(wh) >= 2 else (0, 0)
    if not (sw and sh):
        raise RuntimeError("could not probe source dimensions")
    out_w, out_h, scale_used = _resolve_output_size(
        sw, sh, scale=final_scale, target_height=target_height)
    encoder = "h264_nvenc" if _nvenc_available() else "libx264"  # bounded at 30s + 60s inside _probe_nvenc
    deadline.check("nvenc-probe")
    fsize = sw * sh * 3

    # --- decode (no disk): pull raw rgb24 frames from an ffmpeg pipe into memory ---
    t0 = time.monotonic()
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, bufsize=max(fsize, 1 << 20))
    inputs = []
    with _guarded_child(dec, deadline, "decode") as dec_guard:
        try:
            while True:
                buf = _read_exact(dec.stdout, fsize)
                if buf is None:
                    break
                inputs.append(buf)
                deadline.check("decode")
        finally:
            # The read above blocks in the kernel on a stalled decoder, where no check is reached; the
            # watchdog kills the child and the read returns. dec.wait() used to be unbounded here, so a
            # decoder that ignored the closed pipe hung the invocation on the ABORT path -- the guard's
            # own exit route.
            dec.stdout.close()
            _reap(dec, deadline)
    if dec_guard["fired"]:
        raise InvocationExpired(deadline.reason("decode"))
    if not inputs:
        raise RuntimeError("no frames decoded from source")
    t1 = time.monotonic()

    # --- upscale (GPU, batched) -- drop each input batch as it is consumed to bound peak RAM ---
    outputs = []
    tile_min = TILE  # smallest tile any batch settled on; < TILE means the shrink fallback fired (#30)
    next_tile = None  # carry the settled tile into the next batch instead of re-searching from TILE (#98)
    sec_per_frame = None  # settled rate; None until a batch completes without paying an OOM-search
    try:
        for i in range(0, len(inputs), BATCH):
            # After the first settled batch we know this card's rate. Project the REST of the
            # job against the remaining budget and refuse now if it cannot finish (#98).
            # Burning the whole guard then degrading is the same un-upscaled film, just later
            # and more expensive; silent ok:true at the source resolution is the lie.
            if sec_per_frame is not None:
                frames_left = len(inputs) - i
                deadline.project("upscale", sec_per_frame * frames_left, frames_left)
            chunk = inputs[i:i + BATCH]
            frames_np = [np.frombuffer(b, dtype=np.uint8).reshape(sh, sw, 3) for b in chunk]
            t_b0 = time.monotonic()
            outs, tile_used = _upscale_batch(model, frames_np, out_w, out_h, start_tile=next_tile)
            batch_s = time.monotonic() - t_b0
            tile_min = min(tile_min, tile_used)
            # A batch that shrank paid an OOM-search; its seconds-per-frame is not the
            # settled rate and must not project the rest of the job (it would refuse
            # shots the remaining batches would have finished). x4plus at 720p settles
            # 512 -> 256 on the first batch; the second batch is the rate.
            start_for_this = TILE if next_tile is None else next_tile
            if tile_used >= start_for_this:
                sec_per_frame = batch_s / max(len(chunk), 1)
            next_tile = tile_used
            outputs.extend(np.ascontiguousarray(f).tobytes() for f in outs)
            for j in range(i, min(i + BATCH, len(inputs))):
                inputs[j] = None
            deadline.check("upscale")
    finally:
        # RELEASED ON EVERY EXIT PATH, INCLUDING THE TIMEOUT (#98). This used to sit after the loop, so
        # the `raise TimeoutError` above jumped straight over it and the allocator kept its reservation
        # for the LIFE OF THE PROCESS -- measured at ~19.7 GiB of a 20.5 GiB card still held while the
        # container sat idle with the job long finished, against ~2 GiB after a job that COMPLETED. Both
        # doors share one card on the GPU twins, so a co-tenant needing its own CUDA context was left
        # roughly 700 MiB. The leak was not a property of the model or of torch; it was a property of
        # the abort path, and a `finally` is the whole fix.
        #
        # It also does the job it was originally written for: hand the card to the NVENC encoder with
        # room to work. The upscale phase leaves the torch CUDA caching allocator holding
        # reserved-but-free VRAM that a SEPARATE CUDA context (ffmpeg h264_nvenc) cannot use, so on a
        # memory-tight or co-tenanted card the encoder fails to init its input buffers
        # ("CreateInputBuffer failed: out of memory") and the frame pipe breaks. All outputs are already
        # on the CPU here, so releasing the cache is free; the model weights (allocated, not cached)
        # survive.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    del inputs
    t2 = time.monotonic()

    # --- encode (no disk): feed raw rgb24 frames to an ffmpeg pipe -> nvenc/libx264 ---
    enc_cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}",
               "-framerate", fps, "-i", "-"]
    if _has_audio(src, deadline=deadline):
        enc_cmd += ["-i", src, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-shortest"]
    if encoder == "h264_nvenc":
        enc_cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-pix_fmt", "yuv420p"]
    else:
        enc_cmd += ["-c:v", "libx264", "-crf", "19", "-preset", "fast", "-pix_fmt", "yuv420p"]
    enc_cmd += [dst]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)
    rc = None
    with _guarded_child(enc, deadline, "encode") as enc_guard:
        try:
            for fb in outputs:
                deadline.check("encode")
                enc.stdin.write(fb)
        except BrokenPipeError:
            # The watchdog killed the encoder mid-write, which is HOW a blocked write is interrupted.
            # A broken pipe with the watchdog silent is a genuine encoder crash and still fails loud.
            if not enc_guard["fired"]:
                raise
        finally:
            with contextlib.suppress(Exception):
                enc.stdin.close()
            rc = _reap(enc, deadline)
    # EXPIRY IS JUDGED BEFORE rc. A killed encoder exits non-zero, so rc alone reports the guard firing
    # as `encode pipe exited rc=-9` -- an ordinary error, hiding the ceiling from the operator it fired
    # for, and taking the FAILED-envelope route instead of the honest degrade.
    if enc_guard["fired"]:
        raise InvocationExpired(deadline.reason("encode"))
    if rc != 0:
        raise RuntimeError(f"encode pipe exited rc={rc}")
    t3 = time.monotonic()
    return {
        "frames": len(outputs),
        "encoder": encoder,
        "out_w": out_w, "out_h": out_h,
        "scale": scale_used,
        "extract_s": round(t1 - t0, 2),
        "upscale_s": round(t2 - t1, 2),
        "encode_s": round(t3 - t2, 2),
        "budget_s": deadline.seconds,  # the CONFIGURED value this run was held to, never the literal default
        "batch": BATCH, "fp16": bool(HALF and torch.cuda.is_available()),
        "tile": TILE, "tile_min": tile_min, "tile_shrank": tile_min < TILE,
    }


class _GpuSampler(threading.Thread):
    """Polls nvidia-smi in the background so the selftest can report HONEST GPU utilization + VRAM (and
    best-effort encoder utilization) over a real multi-second clip -- proving the pipeline is GPU-bound,
    not a cherry-picked single number."""

    def __init__(self, period=0.5):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self._period = period
        self.samples = []  # list of (gpu_util%, mem_used_mib, enc_util%|None)

    def run(self):
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self._period)

    def _sample_once(self):
        # utilization.gpu (SM %) + memory.used (MiB) are universally valid --query-gpu fields.
        # (utilization.encoder is NOT a --query-gpu field, so encoder util is read separately below.)
        # Any failure is swallowed -- sampling is best effort and never fails the job.
        try:
            p = _run_bounded(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                timeout=5, capture_output=True, text=True)
            row = (p.stdout or "").strip().splitlines()
            if not row:
                return
            parts = [x.strip() for x in row[0].split(",")]
            if len(parts) < 2:
                return
            gpu_util, mem_used = int(float(parts[0])), int(float(parts[1]))
        except Exception:  # noqa: BLE001
            return
        self.samples.append((gpu_util, mem_used, self._enc_util()))

    @staticmethod
    def _enc_util():
        # Encoder-engine utilization is not in --query-gpu; read the Utilization section of `-q`.
        # Returns None if the field is absent on this driver (then it is just omitted from the report).
        try:
            p = _run_bounded(["nvidia-smi", "-q", "-d", "UTILIZATION"],
                             timeout=5, capture_output=True, text=True)
            for ln in (p.stdout or "").splitlines():
                key, sep, val = ln.partition(":")
                if sep and key.strip() == "Encoder":
                    return int(float(val.strip().rstrip("%").strip()))
        except Exception:  # noqa: BLE001
            pass
        return None

    def stop(self):
        self._stop_event.set()

    def stats(self):
        if not self.samples:
            return {"samples": 0}
        g = [s[0] for s in self.samples]
        m = [s[1] for s in self.samples]
        e = [s[2] for s in self.samples if s[2] is not None]
        out = {
            "samples": len(self.samples),
            "gpu_util_max": max(g), "gpu_util_avg": round(sum(g) / len(g), 1),
            "mem_used_max_mib": max(m),
        }
        if e:
            out["enc_util_max"] = max(e)
            out["enc_util_avg"] = round(sum(e) / len(e), 1)
        return out


def _selftest(inp):
    """Deploy verification. With an explicit `model`, run just that one (back-compat). WITHOUT a model,
    SWEEP every shipped model so a heavy model (RealESRGAN_x4plus) is exercised on silicon at a realistic
    frame count, not only the default -- the S24 gap that let an x4plus OOM ship silent (#584 sib). Also
    runs the R2 finish-contract leg (_upscale_r2 download+upload round-trip) so the real bucket path is
    verified, not just the baked-sample path (#26): OPPORTUNISTIC -- it HONEST-SKIPS when R2 creds are
    absent (r2.ok = None, r2.skipped set) and does NOT fail the sweep -- UNLESS the caller passes
    `"r2": true`, which REQUIRES it (absent creds then FAIL). ok is true only when EVERY swept model
    passed AND the R2 leg did not fail. Trigger with {"selftest": true} (+ optional model / scale / r2,
    plus res "WxH" and dur seconds for the generated test clip -- a large res paired with a large
    UPSCALE_TILE drives the #30 tile-shrink on a small card)."""
    try:
        final_scale, target_height = _parse_size_knobs(inp)
    except SizeRequestError as e:
        return {"ok": False, "selftest": True, "error": str(e)}
    r2_requested = bool(inp.get("r2"))
    res, dur = str(inp.get("res", "1280x720")), inp.get("dur", 3)
    requested = inp.get("model")
    if requested:
        result = _selftest_one(str(requested), final_scale, res, dur, target_height=target_height)
        if r2_requested:
            r2 = _selftest_r2(final_scale, str(requested), requested=True, target_height=target_height)
            result["r2"] = r2
            result["ok"] = bool(result.get("ok")) and r2.get("ok") is not False
        return result
    names = list(MODEL_FILES.keys())
    models = {n: _selftest_one(n, final_scale, res, dur, target_height=target_height) for n in names}
    # R2 leg uses the fast model (the round-trip proves the boto3 path, not model weight; the sweep above
    # already exercises the heavy x4plus on silicon).
    r2 = _selftest_r2(final_scale, names[0], requested=r2_requested, target_height=target_height)
    ok = all(m.get("ok") for m in models.values()) and r2.get("ok") is not False  # None (skipped) passes
    out = {"ok": ok, "selftest": True, "swept": names,
           "cuda_available": torch.cuda.is_available(), "models": models, "r2": r2}
    if final_scale is not None:
        out["scale"] = final_scale
    if target_height is not None:
        out["target_height"] = target_height
    return out


def _selftest_one(model_name, final_scale, res="1280x720", dur=3, target_height=None):
    """End-to-end GPU selftest for ONE model at a target scale (NO R2). Loads the model, generates a real
    multi-second clip at `res` (WxH) for `dur` seconds, upscales it, and reports the encoder used, per-phase
    wall-clock, sampled GPU/encoder utilization + peak VRAM, the batch/fp16 settings, and the tile the run
    settled on (tile_min < tile means the #30 shrink fallback fired -- driveable on a small card by pairing
    a large res with a large UPSCALE_TILE). Returns the per-model result dict."""
    out = {"ok": False, "selftest": True, "torch_version": torch.__version__,
           "cuda_available": torch.cuda.is_available()}
    work = tempfile.mkdtemp(prefix="selftest-")
    src, dst = os.path.join(work, "in.mp4"), os.path.join(work, "out.mp4")
    sampler = _GpuSampler()
    try:
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            torch.cuda.reset_peak_memory_stats()
        out["nvenc_available"] = _nvenc_available()
        model = _load_model(model_name)
        out["model"], out["model_scale"] = model_name, getattr(model, "scale", 4)
        gw, gh = _parse_res(res)
        dur = max(1, min(int(dur or 3), 30))
        out["requested_res"], out["requested_dur"] = f"{gw}x{gh}", dur
        # A real multi-second clip (default 720p24 x 3s = 72 frames) so the GPU work + encode are non-trivial.
        gen = _run_bounded(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
             "-i", f"testsrc=size={gw}x{gh}:rate=24:duration={dur}", "-pix_fmt", "yuv420p", src],
            timeout=PROBE_TIMEOUT, capture_output=True, text=True,
        )
        if gen.returncode != 0:
            out["error"] = f"ffmpeg gen failed: {(gen.stderr or '')[-500:]}"
            return out
        out["input_res"] = "x".join(_ffprobe(src, "stream=width,height"))
        sampler.start()
        t0 = time.monotonic()
        info = _upscale_video(model, src, dst, final_scale, target_height=target_height)
        out["wall_s"] = round(time.monotonic() - t0, 2)
        sampler.stop()
        sampler.join(timeout=2)
        if not os.path.exists(dst) or os.path.getsize(dst) == 0:
            out["error"] = "no output produced"
            return out
        out["frames"] = info["frames"]
        out["encoder"] = info["encoder"]
        out["nvenc_used"] = info["encoder"] == "h264_nvenc"
        out["batch"], out["fp16"] = info["batch"], info["fp16"]
        out["tile"], out["tile_min"] = info["tile"], info["tile_min"]
        out["tile_shrank"] = info["tile_shrank"]
        out["phase_s"] = {"extract": info["extract_s"], "upscale": info["upscale_s"],
                          "encode": info["encode_s"]}
        out["gpu_sample"] = sampler.stats()
        if torch.cuda.is_available():
            out["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        out["output_res"] = "x".join(_ffprobe(dst, "stream=width,height"))
        out["output_bytes"] = os.path.getsize(dst)
        out["scale"] = info["scale"]
        out["out_w"], out["out_h"] = info["out_w"], info["out_h"]
        out["ok"] = True
        return out
    except InvocationExpired as e:  # BaseException-derived: the broad handler below cannot see it
        out["error"] = str(e)[:800]  # a selftest reports diagnostics, not a chain degrade
        out["guard_expired"] = True
        return out
    except Exception as e:  # noqa: BLE001 -- a job error is data, returned to the caller
        out["error"] = str(e)[:800]
        return out
    finally:
        sampler.stop()
        shutil.rmtree(work, ignore_errors=True)


# --- R2 mode (the finish-upscale module contract) -------------------------------------------------
# The module sends clip_key/output_key and the endpoint reads/writes the shared bucket itself (mirrors
# vivijure-backend's finish path), so no presigned URLs or R2 creds cross the module wire.
R2_ENDPOINT = os.environ.get("R2_ENDPOINT_URL", "")
# Optional pin for presigned hosts (e.g. ".r2.cloudflarestorage.com"). Empty = skip host-suffix check.
R2_URL_HOST_SUFFIX = os.environ.get("R2_URL_HOST_SUFFIX", "").strip().lower()

# requests / urllib3 exception text embeds the full URL, including the presigned query.
_FULL_URL_QUERY_RE = re.compile(r"(https?://[^\s'\"<>]+)\?[^\s'\"<>]*", re.IGNORECASE)
_LABELED_URL_QUERY_RE = re.compile(r"(url:\s+\S+?)\?[^\s'\"<>]*", re.IGNORECASE)


def _redact_query(text):
    """Strip query strings so presigned tokens never leave the worker in errors or logs."""
    if not text:
        return text
    s = str(text)
    s = _FULL_URL_QUERY_RE.sub(r"\1?[redacted]", s)
    s = _LABELED_URL_QUERY_RE.sub(r"\1?[redacted]", s)
    return s


def _ip_blocked(ip):
    return (ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified)


def _resolve_public_ips(host):
    """Resolve host; return public IPs or raise ValueError with a job-facing message."""
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"URL host does not resolve: {e}") from e
    public, blocked = [], False
    for _fam, _type, _proto, _canon, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if _ip_blocked(ip):
            blocked = True
        else:
            public.append(str(ip))
    if blocked or not public:
        raise ValueError("URL resolves to a blocked address")
    return public


def _url_error(url, what):
    """Refuse non-https / private / link-local / loopback / optional non-R2 host. Returns err str or None.

    Presigned mode otherwise lets any job submitter drive GET/PUT from the GPU worker (SSRF). Resolve
    the hostname and reject blocked address classes; callers must also pass allow_redirects=False and
    connect to a pre-validated IP (see _pinned_https) so DNS cannot rebind between check and fetch."""
    try:
        p = urlparse(str(url or ""))
    except Exception:  # noqa: BLE001 -- malformed URL is a job error, not a crash
        return f"{what}: malformed URL"
    if p.scheme != "https" or not p.hostname:
        return f"{what}: URL must be https with a hostname"
    host = p.hostname.lower()
    if host == "localhost" or host.endswith(".localhost"):
        return f"{what}: URL host is blocked"
    if R2_URL_HOST_SUFFIX:
        suffix = R2_URL_HOST_SUFFIX if R2_URL_HOST_SUFFIX.startswith(".") else f".{R2_URL_HOST_SUFFIX}"
        bare = suffix.lstrip(".")
        if host != bare and not host.endswith(suffix):
            return f"{what}: URL host must end with {R2_URL_HOST_SUFFIX}"
    try:
        _resolve_public_ips(host)
    except ValueError as e:
        return f"{what}: {e}"
    return None


def _pinned_https(method, url, *, timeout, headers=None, data=None, stream=False):
    """HTTPS GET/PUT that resolves once, rejects private addrs, and connects to that IP (DNS-rebinding safe)."""
    from requests.adapters import HTTPAdapter  # deferred: keeps CPU test stubs light

    class _SniAdapter(HTTPAdapter):
        """Keep TLS SNI / hostname verify on the original host while connecting to a pinned IP."""

        def __init__(self, server_hostname, **kwargs):
            self._server_hostname = server_hostname
            super().__init__(**kwargs)

        def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
            pool_kwargs["assert_hostname"] = self._server_hostname
            pool_kwargs["server_hostname"] = self._server_hostname
            return super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)

    p = urlparse(str(url or ""))
    if p.scheme != "https" or not p.hostname:
        raise ValueError("URL must be https with a hostname")
    host = p.hostname.lower()
    ip = _resolve_public_ips(host)[0]
    netloc_host = f"[{ip}]" if ":" in ip else ip
    netloc = f"{netloc_host}:{p.port}" if p.port else netloc_host
    pinned = urlunparse((p.scheme, netloc, p.path or "/", p.params, p.query, ""))
    hdrs = dict(headers or {})
    hdrs["Host"] = host if not p.port else f"{host}:{p.port}"
    session = requests.Session()
    session.mount("https://", _SniAdapter(host))
    return session.request(method, pinned, timeout=timeout, headers=hdrs, data=data,
                           stream=stream, allow_redirects=False)


R2_BUCKET = os.environ.get("R2_BUCKET", "vivijure")


def _r2():
    """The R2 client, with its transport bounds DECLARED rather than defaulted.

    The bytes in and out of the bucket sit OUTSIDE the _upscale_video deadline by design: that budget
    is the GPU/ffmpeg work, and stretching it over a network transfer would make one number mean two
    things. They are bounded here instead, so the whole invocation has a stated ceiling. The botocore
    import is local because botocore is a boto3 transitive dep that the hermetic tests (which stub
    boto3 and monkeypatch this function) do not install."""
    from botocore.config import Config
    return boto3.client(
        "s3", endpoint_url=R2_ENDPOINT, region_name="auto",
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY", ""),
        config=Config(connect_timeout=R2_CONNECT_TIMEOUT, read_timeout=R2_READ_TIMEOUT,
                      retries={"max_attempts": R2_MAX_ATTEMPTS, "mode": "standard"}),
    )


def _key_error(key, what, prefixes=("renders/",)):
    """Validate a job-supplied R2 key against the render key map BEFORE any bucket I/O. Every key
    this module reads or writes lives inside the studio's render tree (see the module docstring),
    so an absolute key, a `..` segment, a backslash, or an out-of-prefix key is a malformed job.
    Refused as data (this handler reports errors, it does not raise): returns the error string,
    or None when the key is fine."""
    k = str(key or "")
    ok = (bool(k) and k == k.strip() and not k.startswith("/") and "\\" not in k
          and ".." not in k.split("/") and k.startswith(tuple(prefixes)))
    return None if ok else f"{what}: R2 key {k!r} must be a plain relative key under {' or '.join(prefixes)}"


def _project_prefix(project):
    """Trusted project segment for shared-bucket tenancy. Mirrors studio finish keys
    (`renders/${project}/...`) -- reject slash/backslash/whitespace so the field cannot widen the prefix."""
    raw = str(project or "")
    p = raw.strip()
    if not p or p != raw or "/" in p or "\\" in p or any(c.isspace() for c in p):
        return None
    return f"renders/{p}/"


def _scoped_key_error(key, what, *, project, prefixes=("renders/",)):
    """Prefix-check plus project tenancy for renders/ keys."""
    err = _key_error(key, what, prefixes=prefixes)
    if err:
        return err
    pref = _project_prefix(project)
    if not pref:
        return f"{what}: project is required for R2 mode"
    if not str(key).startswith(pref):
        return f"{what}: R2 key must be under {pref}"
    return None


def _stamp_sidecar_r2(s3, output_key, output_hash):
    """#583 provenance: write the core-computed param-hash to `<output_key>.hash` AFTER the artifact
    (artifact first, sidecar last -- the only safe order; studio CONTRACT.md 3.3.1). Opaque: write the
    value verbatim, never recompute it. Best-effort: a failed sidecar only disables reuse (the core
    re-runs), it must NEVER fail a good render. No output_hash (legacy core) -> no sidecar."""
    if not output_hash:
        return
    try:
        s3.put_object(Bucket=R2_BUCKET, Key=f"{output_key}.hash",
                      Body=str(output_hash).encode("utf-8"), ContentType="text/plain")
    except Exception:  # noqa: BLE001 -- provenance is best-effort; a miss = safe re-run, never a failed render
        pass


def _stamp_sidecar_presigned(hash_url, output_hash):
    """Presigned-mode sidecar stamp: the credentialless handler writes the `.hash` only if the core
    presigned a `hash_url`. Prod finish uses R2 mode (this is a no-op there); a presigned deployment gets
    provenance once the core presigns hash_url. Same opaque + best-effort contract."""
    if not (hash_url and output_hash):
        return
    if _url_error(hash_url, "hash_url"):
        return
    try:
        body = str(output_hash).encode("utf-8")
        _pinned_https(
            "PUT", hash_url, timeout=UPLOAD_TIMEOUT, data=body,
            headers={"content-type": "text/plain", "content-length": str(len(body))},
        ).raise_for_status()
    except Exception:  # noqa: BLE001 -- best-effort provenance; a miss = safe re-run
        pass


def _upscale_r2(inp):
    """R2 mode: download clip_key, upscale, upload output_key in the shared bucket; return the new key as
    `clip_key` so the finish chain passes the upscaled clip downstream."""
    clip_key = inp.get("clip_key")
    project = inp.get("project")
    err = _scoped_key_error(clip_key, "clip_key", project=project)
    if err:
        return {"ok": False, "error": err}
    name = clip_key.rsplit("/", 1)[-1]
    output_key = inp.get("output_key") or (
        f"{clip_key.rsplit('.', 1)[0]}_up.{clip_key.rsplit('.', 1)[1]}" if "." in name else f"{clip_key}_up")
    err = _scoped_key_error(output_key, "output_key", project=project)
    if err:
        return {"ok": False, "error": err}
    try:
        final_scale, target_height = _parse_size_knobs(inp)
    except SizeRequestError as e:
        return {"ok": False, "error": str(e)}
    model_name = str(inp.get("model", "realesr-animevideov3"))
    if not (R2_ENDPOINT and os.environ.get("R2_ACCESS_KEY_ID")):
        return {"ok": False, "error": "R2 mode needs R2_ENDPOINT_URL + R2_ACCESS_KEY_ID/SECRET in the endpoint env"}
    work = tempfile.mkdtemp(prefix="up-")
    src, dst = os.path.join(work, "in.mp4"), os.path.join(work, "out.mp4")
    try:
        s3 = _r2()
        s3.download_file(R2_BUCKET, clip_key, src)
        model = _load_model(model_name)
        info = _upscale_video(model, src, dst, final_scale, target_height=target_height)
        if not os.path.getsize(dst):
            return {"ok": False, "error": "upscale produced no output"}
        s3.upload_file(dst, R2_BUCKET, output_key, ExtraArgs={"ContentType": "video/mp4"})
        _stamp_sidecar_r2(s3, output_key, inp.get("output_hash"))  # #583: sidecar AFTER the artifact
        return {"ok": True, "clip_key": output_key, "bytes": os.path.getsize(dst),
                "scale": info["scale"], "out_w": info["out_w"], "out_h": info["out_h"],
                "model": model_name, "frames": info["frames"],
                "encoder": info["encoder"],
                "applied": [_applied_tag(info["out_h"], info["scale"],
                                         from_target_height=target_height is not None)]}
    except InvocationExpired as e:
        # THE SOFT-DEGRADE SHAPE, and both halves of it are load-bearing (#105).
        # RETURN, NEVER RAISE: a raise leaves the RunPod envelope with no structured output, which
        # vivijure-cf's discriminator correctly refuses to absorb, so the whole FILM fails after the GPU
        # spend is already banked -- strictly worse than the hang this guard replaces, since the hang is
        # at least recoverable by the core's phase ceiling.
        # `detail`, NOT `error`: degradeReason reads `detail` first (vivijure-cf
        # modules/_shared/finish-soft-degrade.ts:77), and a top-level `error` is lifted by RunPod into a
        # FAILED envelope which books a `failed` job row for a job that degraded honestly. Ordinary
        # errors below keep `error` on purpose -- they ARE failures, and the legacy shape is still
        # recovered by the panel's FAILED-envelope route.
        return {"ok": False, "detail": _redact_query(str(e)[:120])}
    except Exception as e:  # noqa: BLE001 -- a job error is data, returned to the caller
        return {"ok": False, "error": _redact_query(str(e)[:500])}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _selftest_r2(final_scale, model_name, requested, target_height=None):
    """Exercise the REAL _upscale_r2 finish contract (boto3 download + upload against the shared bucket):
    generate a tiny clip, upload it under a temp renders/ key, run _upscale_r2, confirm the output object
    landed, then delete both objects (and the .hash sidecar). HONEST-FAILURES: if R2 creds/env are absent
    the leg reports {"ok": None, "skipped": "no creds"} and does NOT fail the sweep -- UNLESS the caller
    explicitly asked for it (`requested`), in which case absent creds are a FAILURE (ok False). Returns the
    per-leg result dict."""
    have_creds = bool(R2_ENDPOINT and os.environ.get("R2_ACCESS_KEY_ID")
                      and os.environ.get("R2_SECRET_ACCESS_KEY"))
    if not have_creds:
        if requested:
            return {"ok": False, "requested": True,
                    "error": "R2 leg requested but R2_ENDPOINT_URL + R2_ACCESS_KEY_ID/SECRET are not set"}
        return {"ok": None, "skipped": "no creds"}
    tag = f"{os.getpid()}-{int(time.time())}"
    clip_key = f"renders/_selftest/upscale-{tag}.mp4"
    output_key = f"renders/_selftest/upscale-{tag}_up.mp4"
    work = tempfile.mkdtemp(prefix="selftest-r2-")
    src = os.path.join(work, "in.mp4")
    s3 = _r2()
    leg = {"ok": False, "clip_key": clip_key, "output_key": output_key, "bucket": R2_BUCKET}
    try:
        gen = _run_bounded(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
             "-i", "testsrc=size=320x240:rate=12:duration=1", "-pix_fmt", "yuv420p", src],
            timeout=PROBE_TIMEOUT, capture_output=True, text=True,
        )
        if gen.returncode != 0:
            leg["error"] = f"ffmpeg gen failed: {(gen.stderr or '')[-300:]}"
            return leg
        s3.upload_file(src, R2_BUCKET, clip_key, ExtraArgs={"ContentType": "video/mp4"})
        r2_job = {"project": "_selftest", "clip_key": clip_key, "output_key": output_key,
                  "model": model_name}
        if final_scale is not None:
            r2_job["scale"] = final_scale
        if target_height is not None:
            r2_job["target_height"] = target_height
        res = _upscale_r2(r2_job)
        if not res.get("ok"):
            # `detail` is the guard's degrade key; without it a guard expiry here reads as an
            # unexplained not-ok and the selftest loses the one fact that mattered.
            leg["error"] = res.get("error") or res.get("detail") or "_upscale_r2 returned not-ok"
            return leg
        head = s3.head_object(Bucket=R2_BUCKET, Key=output_key)  # prove the object actually landed
        leg["ok"] = True
        leg["output_bytes"] = head.get("ContentLength")
        leg["encoder"] = res.get("encoder")
        leg["frames"] = res.get("frames")
        leg["model"] = model_name
        return leg
    except Exception as e:  # noqa: BLE001 -- a job error is data, returned to the caller
        leg["error"] = _redact_query(str(e)[:500])
        return leg
    finally:
        # delete the test objects + any provenance sidecar (best effort -- never mask a real result)
        for k in (clip_key, output_key, f"{output_key}.hash"):
            try:
                s3.delete_object(Bucket=R2_BUCKET, Key=k)
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(work, ignore_errors=True)


def handler(job):
    inp = (job or {}).get("input") or {}
    if inp.get("selftest"):
        return _selftest(inp)
    if inp.get("clip_key"):
        return _upscale_r2(inp)
    video_url = inp.get("video_url")
    output_url = inp.get("output_url")
    output_key = inp.get("output_key", "")
    if not video_url or not output_url:
        return {"ok": False, "error": "input needs presigned video_url + output_url"}
    try:
        final_scale, target_height = _parse_size_knobs(inp)
    except SizeRequestError as e:
        return {"ok": False, "error": str(e)}
    for u, name in ((video_url, "video_url"), (output_url, "output_url"),
                    (inp.get("hash_url"), "hash_url")):
        if name == "hash_url" and not u:
            continue
        err = _url_error(u, name)
        if err:
            return {"ok": False, "error": err}
    model_name = str(inp.get("model", "realesr-animevideov3"))
    work = tempfile.mkdtemp(prefix="up-")
    src, dst = os.path.join(work, "in.mp4"), os.path.join(work, "out.mp4")
    try:
        with _pinned_https("GET", video_url, timeout=DOWNLOAD_TIMEOUT, stream=True) as r:
            r.raise_for_status()
            with open(src, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        model = _load_model(model_name)
        info = _upscale_video(model, src, dst, final_scale, target_height=target_height)
        size = os.path.getsize(dst)
        if not size:
            return {"ok": False, "error": "upscale produced no output"}
        with open(dst, "rb") as f:
            put = _pinned_https(
                "PUT", output_url, timeout=UPLOAD_TIMEOUT, data=f,
                headers={"content-type": "video/mp4", "content-length": str(size)})
        put.raise_for_status()
        _stamp_sidecar_presigned(inp.get("hash_url"), inp.get("output_hash"))  # #583: sidecar AFTER the artifact
        return {"ok": True, "output_key": output_key, "bytes": size,
                "scale": info["scale"], "out_w": info["out_w"], "out_h": info["out_h"],
                "model": model_name, "frames": info["frames"],
                "encoder": info["encoder"],
                "applied": [_applied_tag(info["out_h"], info["scale"],
                                         from_target_height=target_height is not None)]}
    except InvocationExpired as e:  # same soft-degrade contract as the R2 path above
        return {"ok": False, "detail": _redact_query(str(e)[:120])}
    except Exception as e:  # noqa: BLE001 -- a job error is data, returned to the caller
        return {"ok": False, "error": _redact_query(str(e)[:500])}
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
