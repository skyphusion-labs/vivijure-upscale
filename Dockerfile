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
# Encode: the handler uses h264_nvenc (hardware encode) on the GPU; the endpoint runs on the 24/48 GB
# PRO (Ada/Ampere) tier, where ffmpeg 4.4's NVENC is well-supported. The build asserts h264_nvenc is
# compiled into ffmpeg below; the on-card encode is proven by the {"selftest": true} verify job.
#
# Base: RunPod's torch 2.8.0 / CUDA 12.8.1 image (ubuntu 22.04, conda), matching the recovered image's
# PyTorch/CUDA stack. torch is provided by the base, so requirements.txt does not pin it. The FROM is
# digest-pinned (not the mutable tag) so a rebuild is deterministic, matching the backend posture.
# tag: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2204 (manifest-list digest, pinned #27). Dockerfile
# syntax has no inline comments, so the tag is recorded here, on a comment line above the FROM.
FROM runpod/pytorch@sha256:263d4144a3053f5125b04174e279d73b43768c5b798cd76c4871af7b737f0c84

ENV DEBIAN_FRONTEND=noninteractive

# ffmpeg/ffprobe from Ubuntu 22.04 apt (4.4.x). The build FAILS here if h264_nvenc is not compiled in,
# so a non-NVENC ffmpeg can never ship silently (the encode path depends on it; libx264 is only a
# runtime fallback). NVENC encoders dlopen the driver's libnvidia-encode at runtime on the GPU host.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Fail the build now if h264_nvenc is not compiled into ffmpeg. No `grep -q`: that closes the pipe on
# first match and ffmpeg dies of SIGPIPE (141) under bash pipefail, failing the build on a SUCCESS. grep
# without -q reads to EOF, so ffmpeg exits 0; the matched encoder line is echoed into the build log.
RUN ffmpeg -hide_banner -encoders 2>/dev/null | grep h264_nvenc

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
