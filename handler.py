"""RunPod serverless handler -- Real-ESRGAN (CUDA) video upscaling for Vivijure's `upscale` module (#191).

Replaces the video2x/Vulkan path (RunPod has no working Vulkan stack -- proven 2026-06-20). Same MODELS
(Real-ESRGAN), run through PyTorch/CUDA via spandrel. The transport contract + the {"selftest": true}
harness are UNCHANGED from the Vulkan attempt -- only the engine swapped.

Job input:
  {
    "video_url":  "<presigned R2 GET of the source clip>",   # required
    "output_url": "<presigned R2 PUT for the result>",       # required
    "output_key": "renders/<project>/clips/<shot>_up.mp4",   # echoed back
    "scale":      2,                          # final factor 2|4 (model is 4x; 2 = 4x then downscale /2)
    "model":      "realesr-animevideov3"      # realesr-animevideov3 (anime/fast) | RealESRGAN_x4plus (general)
  }
Returns: { ok, output_key, bytes, scale, model, frames } on success; { ok: false, error } otherwise.
The upscale module treats a non-ok result as a soft-degrade (passthrough the original) -- never a drop.
"""

import os
import shutil
import subprocess
import tempfile

import boto3
import numpy as np
import requests
import runpod
import torch
from PIL import Image
from spandrel import ModelLoader

MODELS_DIR = "/models"
MODEL_FILES = {
    "realesr-animevideov3": "realesr-animevideov3.pth",
    "RealESRGAN_x4plus": "RealESRGAN_x4plus.pth",
}
DOWNLOAD_TIMEOUT = 900
UPLOAD_TIMEOUT = 900
TILE = 512      # tile size (px) -- bounds GPU memory on large frames
TILE_PAD = 16   # tile overlap to hide seams

_MODELS = {}  # name -> loaded spandrel descriptor (warm-worker cache)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(name):
    name = name if name in MODEL_FILES else "realesr-animevideov3"
    if name not in _MODELS:
        m = ModelLoader().load_from_file(os.path.join(MODELS_DIR, MODEL_FILES[name]))
        m.to(_device()).eval()  # fp32 -- tiling bounds memory; fp16 is a later optimization
        _MODELS[name] = m
    return _MODELS[name]


@torch.inference_mode()
def _upscale_image(model, img):
    """Upscale a PIL image by model.scale, tiled to bound GPU memory; returns a PIL image."""
    scale = getattr(model, "scale", 4)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(_device())
    _, _, h, w = t.shape
    out = torch.zeros((1, 3, h * scale, w * scale), dtype=t.dtype, device=t.device)
    for y in range(0, h, TILE):
        for x in range(0, w, TILE):
            y0, x0 = max(y - TILE_PAD, 0), max(x - TILE_PAD, 0)
            y1, x1 = min(y + TILE + TILE_PAD, h), min(x + TILE + TILE_PAD, w)
            ot = model(t[:, :, y0:y1, x0:x1])  # spandrel descriptor is callable: (1,3,h,w)->(1,3,h*s,w*s)
            cy1, cx1 = min(y + TILE, h), min(x + TILE, w)
            sy, sx = (y - y0) * scale, (x - x0) * scale
            th, tw = (cy1 - y) * scale, (cx1 - x) * scale
            out[:, :, y * scale:cy1 * scale, x * scale:cx1 * scale] = ot[:, :, sy:sy + th, sx:sx + tw]
    a = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((a * 255.0 + 0.5).astype(np.uint8))


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


def _upscale_video(model, src, dst, final_scale):
    """Extract frames -> upscale each (model native scale) -> re-encode, resized to `final_scale` of the
    source. Audio is copied through when the source has it (silent clips just skip it)."""
    work = os.path.dirname(dst)
    fin, fout = os.path.join(work, "fin"), os.path.join(work, "fout")
    os.makedirs(fin, exist_ok=True)
    os.makedirs(fout, exist_ok=True)
    fps = (_ffprobe(src, "stream=r_frame_rate") or ["24/1"])[0]
    wh = _ffprobe(src, "stream=width,height")
    sw, sh = (int(wh[0]), int(wh[1])) if len(wh) >= 2 else (0, 0)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src,
                    os.path.join(fin, "%06d.png")], check=True)
    files = sorted(f for f in os.listdir(fin) if f.endswith(".png"))
    if not files:
        raise RuntimeError("no frames extracted from source")
    for f in files:
        _upscale_image(model, Image.open(os.path.join(fin, f))).save(os.path.join(fout, f))
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-framerate", fps, "-i", os.path.join(fout, "%06d.png")]
    if _has_audio(src):
        cmd += ["-i", src, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-shortest"]
    if sw and sh:
        cmd += ["-vf", f"scale={sw * final_scale}:{sh * final_scale}:flags=lanczos"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "medium", dst]
    subprocess.run(cmd, check=True)
    return len(files)


def _selftest(inp):
    """Self-contained GPU verification -- NO R2 needed. Confirms CUDA + loads the model, then generates a
    tiny clip and actually upscales it end to end. Trigger with {"selftest": true}; doubles as a health check."""
    model_name = str(inp.get("model", "realesr-animevideov3"))
    final_scale = 4 if int(inp.get("scale", 2) or 2) >= 4 else 2
    out = {"ok": False, "selftest": True, "torch_version": torch.__version__,
           "cuda_available": torch.cuda.is_available()}
    work = tempfile.mkdtemp(prefix="selftest-")
    src, dst = os.path.join(work, "in.mp4"), os.path.join(work, "out.mp4")
    try:
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
        model = _load_model(model_name)
        out["model"], out["model_scale"] = model_name, getattr(model, "scale", 4)
        gen = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
             "-i", "testsrc=size=320x240:rate=10:duration=1", "-pix_fmt", "yuv420p", src],
            capture_output=True, text=True,
        )
        if gen.returncode != 0:
            out["error"] = f"ffmpeg gen failed: {(gen.stderr or '')[-500:]}"
            return out
        out["input_res"] = "x".join(_ffprobe(src, "stream=width,height"))
        out["frames"] = _upscale_video(model, src, dst, final_scale)
        if not os.path.exists(dst) or os.path.getsize(dst) == 0:
            out["error"] = "no output produced"
            return out
        out["output_res"] = "x".join(_ffprobe(dst, "stream=width,height"))
        out["output_bytes"] = os.path.getsize(dst)
        out["scale"] = final_scale
        out["ok"] = True
        return out
    except Exception as e:  # noqa: BLE001 -- a job error is data, returned to the caller
        out["error"] = str(e)[:800]
        return out
    finally:
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
        frames = _upscale_video(model, src, dst, final_scale)
        if not os.path.getsize(dst):
            return {"ok": False, "error": "upscale produced no output"}
        s3.upload_file(dst, R2_BUCKET, output_key, ExtraArgs={"ContentType": "video/mp4"})
        return {"ok": True, "clip_key": output_key, "bytes": os.path.getsize(dst),
                "scale": final_scale, "model": model_name, "frames": frames,
                "applied": [f"upscale:{final_scale}x"]}
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
        frames = _upscale_video(model, src, dst, final_scale)
        size = os.path.getsize(dst)
        if not size:
            return {"ok": False, "error": "upscale produced no output"}
        with open(dst, "rb") as f:
            put = requests.put(output_url, data=f, timeout=UPLOAD_TIMEOUT,
                               headers={"content-type": "video/mp4", "content-length": str(size)})
        put.raise_for_status()
        return {"ok": True, "output_key": output_key, "bytes": size,
                "scale": final_scale, "model": model_name, "frames": frames}
    except Exception as e:  # noqa: BLE001 -- a job error is data, returned to the caller
        return {"ok": False, "error": str(e)[:500]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})
