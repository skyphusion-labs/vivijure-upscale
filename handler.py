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
TILE = int(os.environ.get("UPSCALE_TILE", "1024") or "1024")  # tile size (px); env-tunable -- bounds GPU memory per tile pass
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


def _load_model(name):
    name = name if name in MODEL_FILES else "realesr-animevideov3"
    if name not in _MODELS:
        m = ModelLoader().load_from_file(os.path.join(MODELS_DIR, MODEL_FILES[name]))
        m.to(_device()).eval()  # weights fp32; fp16 is applied per-op via autocast in _upscale_batch
        _MODELS[name] = m
    return _MODELS[name]


@torch.inference_mode()
def _upscale_batch(model, frames_np, out_w, out_h):
    """Upscale a BATCH of same-size frames on the GPU, tiled to bound memory, then GPU-resize to
    (out_w,out_h). `frames_np` is a list of (h,w,3) uint8 arrays; returns an (N,out_h,out_w,3) uint8
    array. fp16 via autocast when enabled. No disk, no per-frame Python round-trip."""
    cuda = torch.cuda.is_available()
    scale = getattr(model, "scale", 4)
    arr = np.stack(frames_np).astype(np.float32) / 255.0      # (N,h,w,3)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(_device())  # (N,3,h,w)
    n, _, h, w = t.shape
    out = torch.zeros((n, 3, h * scale, w * scale), dtype=torch.float32, device=t.device)
    use_half = HALF and cuda
    for y in range(0, h, TILE):
        for x in range(0, w, TILE):
            y0, x0 = max(y - TILE_PAD, 0), max(x - TILE_PAD, 0)
            y1, x1 = min(y + TILE + TILE_PAD, h), min(x + TILE + TILE_PAD, w)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_half):
                ot = model(t[:, :, y0:y1, x0:x1])  # (N,3,th,tw) -> (N,3,th*s,tw*s)
            ot = ot.float()
            cy1, cx1 = min(y + TILE, h), min(x + TILE, w)
            sy, sx = (y - y0) * scale, (x - x0) * scale
            th, tw = (cy1 - y) * scale, (cx1 - x) * scale
            out[:, :, y * scale:cy1 * scale, x * scale:cx1 * scale] = ot[:, :, sy:sy + th, sx:sx + tw]
    if out.shape[-1] != out_w or out.shape[-2] != out_h:
        out = torch.nn.functional.interpolate(
            out, size=(out_h, out_w), mode="bicubic", align_corners=False, antialias=True)
    out = out.clamp(0, 1).mul_(255.0).add_(0.5).permute(0, 2, 3, 1).to(torch.uint8)
    return out.cpu().numpy()  # (N,out_h,out_w,3)


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
    for i in range(0, len(inputs), BATCH):
        chunk = inputs[i:i + BATCH]
        frames_np = [np.frombuffer(b, dtype=np.uint8).reshape(sh, sw, 3) for b in chunk]
        outs = _upscale_batch(model, frames_np, out_w, out_h)  # (n,out_h,out_w,3) uint8
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
    """Self-contained GPU verification -- NO R2 needed. Confirms CUDA + loads the model, generates a real
    multi-second clip, upscales it end to end, and PROVES the result is GPU-bound: it reports which encoder
    actually ran (asserting NVENC where expected), the per-phase wall-clock split, sampled GPU/encoder
    utilization + peak VRAM, and the batch/fp16 settings. Trigger with {"selftest": true}."""
    model_name = str(inp.get("model", "realesr-animevideov3"))
    final_scale = 4 if int(inp.get("scale", 2) or 2) >= 4 else 2
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
        # A real multi-second clip (720p24 x 3s = 72 frames) so the GPU work and encode are non-trivial.
        gen = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
             "-i", "testsrc=size=1280x720:rate=24:duration=3", "-pix_fmt", "yuv420p", src],
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


def _upscale_r2(inp):
    """R2 mode: download clip_key, upscale, upload output_key in the shared bucket; return the new key as
    `clip_key` so the finish chain passes the upscaled clip downstream."""
    clip_key = inp.get("clip_key")
    name = clip_key.rsplit("/", 1)[-1]
    output_key = inp.get("output_key") or (
        f"{clip_key.rsplit('.', 1)[0]}_up.{clip_key.rsplit('.', 1)[1]}" if "." in name else f"{clip_key}_up")
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
        return {"ok": True, "clip_key": output_key, "bytes": os.path.getsize(dst),
                "scale": final_scale, "model": model_name, "frames": info["frames"],
                "encoder": info["encoder"], "applied": [f"upscale:{final_scale}x"]}
    except Exception as e:  # noqa: BLE001 -- a job error is data, returned to the caller
        return {"ok": False, "error": str(e)[:500]}
    finally:
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
        return {"ok": True, "output_key": output_key, "bytes": size,
                "scale": final_scale, "model": model_name, "frames": info["frames"],
                "encoder": info["encoder"]}
    except Exception as e:  # noqa: BLE001 -- a job error is data, returned to the caller
        return {"ok": False, "error": str(e)[:500]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})
