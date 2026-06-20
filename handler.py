"""RunPod serverless handler for video2x upscaling (Vivijure `upscale` module, #191).

Transport mirrors the rest of the Vivijure pipeline: the core presigns a GET for the source clip and
a PUT for the result; this handler downloads, runs the `video2x` CLI, and uploads the upscaled clip.
The handler holds NO R2 credentials -- only the presigned URLs the caller hands it.

Job input:
  {
    "video_url":  "<presigned R2 GET of the source clip>",   # required
    "output_url": "<presigned R2 PUT for the result>",       # required
    "output_key": "renders/<project>/clips/<shot>_up.mp4",   # echoed back
    "scale":      2,                  # 2 | 4
    "processor":  "realesrgan",       # realesrgan | libplacebo
    "model":      "realesr-animevideov3"   # realesrgan model name
  }
Returns: { ok, output_key, bytes, scale, processor } on success; { ok: false, error } otherwise.
The module treats a non-ok result as a soft-degrade (passthrough the original clip) -- never a drop.
"""

import os
import subprocess
import tempfile

import requests
import runpod

DOWNLOAD_TIMEOUT = 900
UPLOAD_TIMEOUT = 900


def _run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        # surface the tail of stderr; video2x is chatty on Vulkan/model errors
        raise RuntimeError(f"video2x exit {p.returncode}: {(p.stderr or '')[-1000:]}")


def handler(job):
    inp = (job or {}).get("input") or {}
    video_url = inp.get("video_url")
    output_url = inp.get("output_url")
    output_key = inp.get("output_key", "")
    if not video_url or not output_url:
        return {"ok": False, "error": "input needs presigned video_url + output_url"}

    try:
        scale = int(inp.get("scale", 2))
    except (TypeError, ValueError):
        scale = 2
    scale = 4 if scale >= 4 else 2  # video2x integer factors; clamp to 2/4
    processor = str(inp.get("processor", "realesrgan"))
    model = str(inp.get("model", "realesr-animevideov3"))

    work = tempfile.mkdtemp(prefix="up-")
    src = os.path.join(work, "in.mp4")
    dst = os.path.join(work, "out.mp4")
    try:
        # 1. download the source clip (presigned GET)
        with requests.get(video_url, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
            r.raise_for_status()
            with open(src, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)

        # 2. upscale: video2x -i in -o out -s <scale> -p <processor> [--realesrgan-model <model>]
        cmd = ["video2x", "-i", src, "-o", dst, "-s", str(scale), "-p", processor]
        if processor == "realesrgan":
            cmd += ["--realesrgan-model", model]
        _run(cmd)
        if not os.path.exists(dst) or os.path.getsize(dst) == 0:
            return {"ok": False, "error": "video2x produced no output"}

        # 3. upload the result (presigned PUT)
        size = os.path.getsize(dst)
        with open(dst, "rb") as f:
            put = requests.put(
                output_url, data=f, timeout=UPLOAD_TIMEOUT,
                headers={"content-type": "video/mp4", "content-length": str(size)},
            )
        put.raise_for_status()
        return {"ok": True, "output_key": output_key, "bytes": size, "scale": scale, "processor": processor}
    except Exception as e:  # noqa: BLE001 -- a job error is data, returned to the caller
        return {"ok": False, "error": str(e)[:500]}
    finally:
        for p in (src, dst):
            try:
                os.path.exists(p) and os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(work)
        except OSError:
            pass


runpod.serverless.start({"handler": handler})
