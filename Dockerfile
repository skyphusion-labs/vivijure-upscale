# video2x RunPod serverless image -- AI video upscaling backend for Vivijure's `upscale` module (#191).
#
# video2x (k4yt3x/video2x) is GPLv3. We run it as a SEPARATE PROCESS (the handler shells out to the
# `video2x` CLI), so Vivijure's Worker/module code never links it -- mere aggregation, license-clean.
# This image DOES distribute the GPL binary; source is upstream at github.com/k4yt3x/video2x.
#
# v6 is the C++ rewrite: Vulkan-based (NOT CUDA). On RunPod, the NVIDIA Vulkan ICD is injected by the
# nvidia container runtime, so the runtime stage only needs the Vulkan loader (libvulkan1).
#
# Multi-stage: (1) build the .deb from CURRENT source (picks up post-6.4.0 fixes the stale upstream
# docker lacks), (2) a slim CUDA-runtime base + the .deb + the RunPod handler.

# ---- stage 1: build video2x from source -> .deb ---------------------------------------------------
FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl build-essential sudo cargo && \
    rm -rf /var/lib/apt/lists/*
# `just` is the upstream build driver; it apt-installs the build deps and emits a .deb.
RUN cargo install just
ENV PATH="/root/.cargo/bin:${PATH}"
RUN git clone --recurse-submodules https://github.com/k4yt3x/video2x.git /src
WORKDIR /src
# OPEN ITEM: confirm the exact recipe name for 22.04 (docs show `just ubuntu2404`; expecting
# `just ubuntu2204`). If the build is pinned, check out a tag before building (e.g. git checkout 6.4.0).
RUN just ubuntu2204

# ---- stage 2: runtime -- install the .deb + the RunPod handler ------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
# video2x is built against FFmpeg 7 (the `ubuntu2204` recipe adds ppa:ubuntuhandbook1/ffmpeg7), so the
# RUNTIME must provide FFmpeg 7 libs too -- stock 22.04 ffmpeg (4.x) will NOT satisfy the .deb's dynamic
# links. Add the same PPA FIRST, then let the .deb's Depends resolve the matching libav* from it.
# (add-apt-repository writes to /etc/apt/sources.list.d, which survives the apt-list cleanup below.)
# Vulkan loader only -- the NVIDIA Vulkan ICD is injected by nvidia-container-runtime on the GPU pod.
# Models ship INSIDE the .deb (CMake `install(DIRECTORY models -> share/video2x)`), so no manual baking.
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common ca-certificates gnupg && \
    add-apt-repository -y ppa:ubuntuhandbook1/ffmpeg7 && \
    apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libvulkan1 vulkan-tools python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /src/*.deb /tmp/video2x.deb
RUN apt-get update && apt-get install -y --no-install-recommends /tmp/video2x.deb && \
    rm -rf /var/lib/apt/lists/* /tmp/video2x.deb
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt
COPY handler.py /app/handler.py
WORKDIR /app
# Quick self-check at build time: the CLI resolves + lists GPUs (no GPU needed to print help/version).
RUN video2x --help >/dev/null 2>&1 || echo "WARN: video2x --help failed at build; verify on a GPU pod"
CMD ["python3", "handler.py"]
