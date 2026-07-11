"""RunPod serverless handler -- Real-ESRGAN (CUDA) video upscaling for Vivijure's `upscale` module (#191).

Replaces the video2x/Vulkan path (RunPod has no working Vulkan stack -- proven 2026-06-20). Same MODELS
(Real-ESRGAN), run through PyTorch/CUDA via spandrel. The transport contract + the {"selftest": true}
harness are UNCHANGED from the Vulkan attempt -- only the engine swapped.

The pipeline is GPU-bound end to end: frames are streamed through ffmpeg pipes (raw rgb24 in and out --
NO per-frame PNG disk roundtrip), upscaled in BATCHES on the GPU (fp16 via autocast), the final-size
rescale runs on the GPU, and the re-encode uses NVENC (`h264_nvenc`) when the card + ffmpeg support it.
The output resolution is clamped (model is 4x native, but a 2x request is rescaled to 2x on the GPU, not
4x-then-CPU-downscale) and the long edge is capped (default 2160p / 4K UHD). If NVENC is not usable on
this image, encode HONESTLY falls back to a bounded libx264 (the resolution cap keeps the CPU encode
bounded); the chosen encoder is reported in the result so a fallback is never silent.

Job input:
  {
    "video_url":  "<presigned R2 GET of the source clip>",   # required (presigned mode)
    "output_url": "<presigned R2 PUT for the result>",       # required (presigned mode)
    "output_key": "renders/<project>/clips/<shot>_up.mp4",   # echoed back
    "scale":      2,                          # final factor 2|4 (model is 4x; 2 = 4x then GPU downscale /2)
    "model":      "realesr-animevideov3"      # realesr-animevideov3 (anime/fast) | RealESRGAN_x4plus (general)
  }
Returns: { ok, output_key, bytes, scale, model, frames, encoder } on success; { ok: false, error }
otherwise. The upscale module treats a non-ok result as a soft-degrade (passthrough the original) --
never a drop. Every ffmpeg phase is wall-clock guarded so a pathological clip degrades instead of hanging.
"""

import os

# PYTORCH_CUDA_ALLOC_CONF is read ONCE by torch at import to configure the CUDA caching allocator, so it
# must be set before `import torch` (spandrel/torch below pull it in). expandable_segments:True lets the
# allocator grow and release segments instead of stranding reserved-but-unallocated VRAM as fragmentation
# -- the ~51 GiB reserved-but-free that filled the card at the x4plus OOM (#30). setdefault so an
# operator-set PYTORCH_CUDA_ALLOC_CONF in the endpoint env always wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import shutil
import subprocess
import tempfile
import threading
import time

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
# Per-phase wall-clock guard (s). A phase that blows past this aborts and the job degrades (ok:false ->
# module passthrough) instead of hanging to the RunPod execution-timeout.
FFMPEG_TIMEOUT = int(os.environ.get("FFMPEG_TIMEOUT", "1200") or "1200")

_MODELS = {}  # name -> loaded spandrel descriptor (warm-worker cache)
_NVENC = None  # tri-state cache: None = unprobed, True/False = h264_nvenc usable on this worker


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _run(cmd, timeout=FFMPEG_TIMEOUT):
    """subprocess.run with a hard wall-clock guard, used for the short probe/utility ffmpeg calls."""
    return subprocess.run(cmd, check=True, timeout=timeout)


def _probe_nvenc():
    """True only if h264_nvenc is BOTH listed AND actually encodes on this GPU. An old ffmpeg NVENC API
    (e.g. an old Ubuntu build) can list the encoder yet fail at runtime on some GPU/driver combos, so a
    real test encode is the only honest check; the chosen encoder is reported. Cached on the warm worker."""
    try:
        enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=30)
        if "h264_nvenc" not in (enc.stdout or ""):
            return False
        test = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "lavfi",
             "-i", "testsrc=size=320x240:rate=10:duration=1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60)
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
def _upscale_batch(model, frames_np, out_w, out_h):
    """Upscale a BATCH of same-size frames on the GPU, tiled to bound memory, then GPU-resize to
    (out_w,out_h). `frames_np` is a list of (h,w,3) uint8 arrays; returns ((N,out_h,out_w,3) uint8 array,
    tile_used). fp16 via autocast when enabled. Starts at TILE and, on a single-frame CUDA OOM, shrinks the
    tile (halving down to TILE_FLOOR) so a small card still finishes (#30). No disk, no per-frame round-trip."""
    cuda = torch.cuda.is_available()
    scale = getattr(model, "scale", 4)
    arr = np.stack(frames_np).astype(np.float32) / 255.0      # (N,h,w,3)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(_device())  # (N,3,h,w)
    use_half = HALF and cuda
    used = {"tile": TILE}  # the tile the successful pass settled on (records a shrink for the report)
    def _pass(tile):
        used["tile"] = tile
        return _tile_pass(model, t, scale, tile, use_half)
    out = _shrink_on_oom(_pass, TILE, TILE_FLOOR,
                         cleanup=(torch.cuda.empty_cache if cuda else None))
    if out.shape[-1] != out_w or out.shape[-2] != out_h:
        out = torch.nn.functional.interpolate(
            out, size=(out_h, out_w), mode="bicubic", align_corners=False, antialias=True)
    out = out.clamp(0, 1).mul_(255.0).add_(0.5).permute(0, 2, 3, 1).to(torch.uint8)
    return out.cpu().numpy(), used["tile"]  # (N,out_h,out_w,3), tile the pass settled on


def _ffprobe(path, entries):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", entries, "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True,
    )
    return [ln for ln in (p.stdout or "").strip().splitlines() if ln]


def _has_audio(path):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
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


def _upscale_video(model, src, dst, final_scale):
    """Decode -> batched GPU upscale + GPU resize -> re-encode, entirely through ffmpeg rawvideo pipes
    (no PNG disk roundtrip). Audio is copied when present. Returns a dict: frames, encoder, out dims,
    per-phase seconds (decode/upscale/encode), and the batch/fp16 settings actually used."""
    fps = (_ffprobe(src, "stream=r_frame_rate") or ["24/1"])[0]
    wh = _ffprobe(src, "stream=width,height")
    sw, sh = (int(wh[0]), int(wh[1])) if len(wh) >= 2 else (0, 0)
    if not (sw and sh):
        raise RuntimeError("could not probe source dimensions")
    out_w, out_h = _capped(sw * final_scale, sh * final_scale, MAX_LONG_EDGE)
    encoder = "h264_nvenc" if _nvenc_available() else "libx264"
    deadline = time.monotonic() + FFMPEG_TIMEOUT
    fsize = sw * sh * 3

    # --- decode (no disk): pull raw rgb24 frames from an ffmpeg pipe into memory ---
    t0 = time.monotonic()
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, bufsize=max(fsize, 1 << 20))
    inputs = []
    try:
        while True:
            buf = _read_exact(dec.stdout, fsize)
            if buf is None:
                break
            inputs.append(buf)
            if time.monotonic() > deadline:
                raise TimeoutError("decode exceeded FFMPEG_TIMEOUT")
    finally:
        dec.stdout.close()
        dec.wait()
    if not inputs:
        raise RuntimeError("no frames decoded from source")
    t1 = time.monotonic()

    # --- upscale (GPU, batched) -- drop each input batch as it is consumed to bound peak RAM ---
    outputs = []
    tile_min = TILE  # smallest tile any batch settled on; < TILE means the shrink fallback fired (#30)
    for i in range(0, len(inputs), BATCH):
        chunk = inputs[i:i + BATCH]
        frames_np = [np.frombuffer(b, dtype=np.uint8).reshape(sh, sw, 3) for b in chunk]
        outs, tile_used = _upscale_batch(model, frames_np, out_w, out_h)  # (n,out_h,out_w,3) uint8, tile
        tile_min = min(tile_min, tile_used)
        outputs.extend(np.ascontiguousarray(f).tobytes() for f in outs)
        for j in range(i, min(i + BATCH, len(inputs))):
            inputs[j] = None
        if time.monotonic() > deadline:
            raise TimeoutError("upscale exceeded FFMPEG_TIMEOUT")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    del inputs
    t2 = time.monotonic()

    # --- encode (no disk): feed raw rgb24 frames to an ffmpeg pipe -> nvenc/libx264 ---
    enc_cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}",
               "-framerate", fps, "-i", "-"]
    if _has_audio(src):
        enc_cmd += ["-i", src, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-shortest"]
    if encoder == "h264_nvenc":
        enc_cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-pix_fmt", "yuv420p"]
    else:
        enc_cmd += ["-c:v", "libx264", "-crf", "19", "-preset", "fast", "-pix_fmt", "yuv420p"]
    enc_cmd += [dst]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)
    try:
        for fb in outputs:
            enc.stdin.write(fb)
    finally:
        enc.stdin.close()
        rc = enc.wait()
    if rc != 0:
        raise RuntimeError(f"encode pipe exited rc={rc}")
    t3 = time.monotonic()
    return {
        "frames": len(outputs),
        "encoder": encoder,
        "out_w": out_w, "out_h": out_h,
        "extract_s": round(t1 - t0, 2),
        "upscale_s": round(t2 - t1, 2),
        "encode_s": round(t3 - t2, 2),
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
            p = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
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
            p = subprocess.run(["nvidia-smi", "-q", "-d", "UTILIZATION"],
                               capture_output=True, text=True, timeout=5)
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
    final_scale = 4 if int(inp.get("scale", 2) or 2) >= 4 else 2
    r2_requested = bool(inp.get("r2"))
    res, dur = str(inp.get("res", "1280x720")), inp.get("dur", 3)
    requested = inp.get("model")
    if requested:
        result = _selftest_one(str(requested), final_scale, res, dur)
        if r2_requested:
            r2 = _selftest_r2(final_scale, str(requested), requested=True)
            result["r2"] = r2
            result["ok"] = bool(result.get("ok")) and r2.get("ok") is not False
        return result
    names = list(MODEL_FILES.keys())
    models = {n: _selftest_one(n, final_scale, res, dur) for n in names}
    # R2 leg uses the fast model (the round-trip proves the boto3 path, not model weight; the sweep above
    # already exercises the heavy x4plus on silicon).
    r2 = _selftest_r2(final_scale, names[0], requested=r2_requested)
    ok = all(m.get("ok") for m in models.values()) and r2.get("ok") is not False  # None (skipped) passes
    return {"ok": ok, "selftest": True, "swept": names, "scale": final_scale,
            "cuda_available": torch.cuda.is_available(), "models": models, "r2": r2}


def _selftest_one(model_name, final_scale, res="1280x720", dur=3):
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
        gen = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
             "-i", f"testsrc=size={gw}x{gh}:rate=24:duration={dur}", "-pix_fmt", "yuv420p", src],
            capture_output=True, text=True,
        )
        if gen.returncode != 0:
            out["error"] = f"ffmpeg gen failed: {(gen.stderr or '')[-500:]}"
            return out
        out["input_res"] = "x".join(_ffprobe(src, "stream=width,height"))
        sampler.start()
        t0 = time.monotonic()
        info = _upscale_video(model, src, dst, final_scale)
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
        out["scale"] = final_scale
        out["ok"] = True
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
R2_BUCKET = os.environ.get("R2_BUCKET", "vivijure")


def _r2():
    return boto3.client(
        "s3", endpoint_url=R2_ENDPOINT, region_name="auto",
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY", ""),
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
    try:
        body = str(output_hash).encode("utf-8")
        requests.put(hash_url, data=body, timeout=UPLOAD_TIMEOUT,
                     headers={"content-type": "text/plain", "content-length": str(len(body))}).raise_for_status()
    except Exception:  # noqa: BLE001 -- best-effort provenance; a miss = safe re-run
        pass


def _upscale_r2(inp):
    """R2 mode: download clip_key, upscale, upload output_key in the shared bucket; return the new key as
    `clip_key` so the finish chain passes the upscaled clip downstream."""
    clip_key = inp.get("clip_key")
    err = _key_error(clip_key, "clip_key")
    if err:
        return {"ok": False, "error": err}
    name = clip_key.rsplit("/", 1)[-1]
    output_key = inp.get("output_key") or (
        f"{clip_key.rsplit('.', 1)[0]}_up.{clip_key.rsplit('.', 1)[1]}" if "." in name else f"{clip_key}_up")
    err = _key_error(output_key, "output_key")
    if err:
        return {"ok": False, "error": err}
    final_scale = 4 if int(inp.get("scale", 2) or 2) >= 4 else 2
    model_name = str(inp.get("model", "realesr-animevideov3"))
    if not (R2_ENDPOINT and os.environ.get("R2_ACCESS_KEY_ID")):
        return {"ok": False, "error": "R2 mode needs R2_ENDPOINT_URL + R2_ACCESS_KEY_ID/SECRET in the endpoint env"}
    work = tempfile.mkdtemp(prefix="up-")
    src, dst = os.path.join(work, "in.mp4"), os.path.join(work, "out.mp4")
    try:
        s3 = _r2()
        s3.download_file(R2_BUCKET, clip_key, src)
        model = _load_model(model_name)
        info = _upscale_video(model, src, dst, final_scale)
        if not os.path.getsize(dst):
            return {"ok": False, "error": "upscale produced no output"}
        s3.upload_file(dst, R2_BUCKET, output_key, ExtraArgs={"ContentType": "video/mp4"})
        _stamp_sidecar_r2(s3, output_key, inp.get("output_hash"))  # #583: sidecar AFTER the artifact
        return {"ok": True, "clip_key": output_key, "bytes": os.path.getsize(dst),
                "scale": final_scale, "model": model_name, "frames": info["frames"],
                "encoder": info["encoder"], "applied": [f"upscale:{final_scale}x"]}
    except Exception as e:  # noqa: BLE001 -- a job error is data, returned to the caller
        return {"ok": False, "error": str(e)[:500]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _selftest_r2(final_scale, model_name, requested):
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
        gen = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
             "-i", "testsrc=size=320x240:rate=12:duration=1", "-pix_fmt", "yuv420p", src],
            capture_output=True, text=True,
        )
        if gen.returncode != 0:
            leg["error"] = f"ffmpeg gen failed: {(gen.stderr or '')[-300:]}"
            return leg
        s3.upload_file(src, R2_BUCKET, clip_key, ExtraArgs={"ContentType": "video/mp4"})
        res = _upscale_r2({"clip_key": clip_key, "output_key": output_key,
                           "scale": final_scale, "model": model_name})
        if not res.get("ok"):
            leg["error"] = res.get("error", "_upscale_r2 returned not-ok")
            return leg
        head = s3.head_object(Bucket=R2_BUCKET, Key=output_key)  # prove the object actually landed
        leg["ok"] = True
        leg["output_bytes"] = head.get("ContentLength")
        leg["encoder"] = res.get("encoder")
        leg["frames"] = res.get("frames")
        leg["model"] = model_name
        return leg
    except Exception as e:  # noqa: BLE001 -- a job error is data, returned to the caller
        leg["error"] = str(e)[:500]
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
    final_scale = 4 if int(inp.get("scale", 2) or 2) >= 4 else 2
    model_name = str(inp.get("model", "realesr-animevideov3"))
    work = tempfile.mkdtemp(prefix="up-")
    src, dst = os.path.join(work, "in.mp4"), os.path.join(work, "out.mp4")
    try:
        with requests.get(video_url, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
            r.raise_for_status()
            with open(src, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        model = _load_model(model_name)
        info = _upscale_video(model, src, dst, final_scale)
        size = os.path.getsize(dst)
        if not size:
            return {"ok": False, "error": "upscale produced no output"}
        with open(dst, "rb") as f:
            put = requests.put(output_url, data=f, timeout=UPLOAD_TIMEOUT,
                               headers={"content-type": "video/mp4", "content-length": str(size)})
        put.raise_for_status()
        _stamp_sidecar_presigned(inp.get("hash_url"), inp.get("output_hash"))  # #583: sidecar AFTER the artifact
        return {"ok": True, "output_key": output_key, "bytes": size,
                "scale": final_scale, "model": model_name, "frames": info["frames"],
                "encoder": info["encoder"]}
    except Exception as e:  # noqa: BLE001 -- a job error is data, returned to the caller
        return {"ok": False, "error": str(e)[:500]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})
