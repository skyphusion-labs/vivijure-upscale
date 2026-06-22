# vivijure-upscale -- Real-ESRGAN (CUDA) video upscaling, RunPod serverless image.
#
# PROVENANCE: this source was recovered from the published image
# ghcr.io/skyphusion-labs/vivijure-upscale:0.2.2 -- the original was built on a RunPod pod that has
# since been terminated and was never committed to version control, so the image was the only
# surviving copy. handler.py and requirements.txt are extracted verbatim from the image; this
# Dockerfile is reconstructed from `docker history` -- functionally faithful and buildable, not
# byte-identical to the lost original.
#
# Engine: Real-ESRGAN run through PyTorch/CUDA via spandrel. This REPLACES the earlier video2x/Vulkan
# attempt (RunPod has no working Vulkan stack -- proven 2026-06-20); same Real-ESRGAN models, CUDA
# engine instead of Vulkan. The transport contract and the {"selftest": true} harness are unchanged.
#
# Base: RunPod's torch 2.8.0 / CUDA 12.8.1 image (ubuntu 22.04, conda), matching the recovered image's
# PyTorch/CUDA stack. torch is provided by the base, so requirements.txt does not pin it.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Bake the Real-ESRGAN weights into the image (no network volume). Pulled from xinntao's public
# GitHub releases (BSD-3-Clause). realesr-animevideov3 = anime/fast; RealESRGAN_x4plus = general.
RUN mkdir -p /models && \
    curl -fsSL -o /models/realesr-animevideov3.pth \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth && \
    curl -fsSL -o /models/RealESRGAN_x4plus.pth \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

COPY handler.py /app/handler.py
WORKDIR /app
CMD ["python", "handler.py"]
