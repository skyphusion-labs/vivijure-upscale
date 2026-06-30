# CLAUDE.md

Guidance for Claude Code (and the crew) working in this repo.

> Default branch is **`master`** (not `main`). Commit and push to `master`.

## What this is

**The GPU backend for Vivijure's `upscale` finish module (#191).** A single RunPod serverless image
that upscales video with **Real-ESRGAN** run on PyTorch/CUDA via
[spandrel](https://github.com/chaiNNer-org/spandrel): stream frames through ffmpeg pipes -> upscale in
BATCHES on the GPU (fp16) -> re-encode (NVENC), audio copied through when present. No per-frame PNG disk
roundtrip; the GPU is the bottleneck, not I/O. Finish-class: it raises the assembled film's resolution,
and is the natural partner that returns a MuseTalk-synced shot to delivery res.

This repo is the image + the RunPod handler; the studio-side `upscale` module worker (a thin CF Worker
behind the typed finish hook in `vivijure`) is what calls this endpoint. Image:
`ghcr.io/skyphusion-labs/vivijure-upscale` (current release tag **v0.2.6**, the immutable tag the
endpoint pins to).

## The Vivijure constellation (the same map is in each repo)

```
   friends + Slate (Discord)
            |
            v
        slate  -->  vivijure (studio control plane / JSON API)
                        |
                        v
                  vivijure-backend (GPU render: keyframes -> i2v -> assemble)
                        |
            +-----------+-------------+-------------------+
            |           |             |                   |
   vivijure-musetalk  vivijure-   vivijure-audio-   vivijure-local-backend
   (lipsync module)   upscale     upscale           (self-host render path)
                      ^-- THIS REPO
```

## Handler contract (the job, `handler.py`)

One typed in / one typed out, three dispatch modes (`handler(job)` branches on the input keys):

- **R2 finish-chain mode** (the endpoint reads/writes the shared bucket itself, no creds on the wire):
  `{ clip_key, output_key?, scale?, model? }`. Returns the new key as `clip_key` so the finish chain
  carries the upscaled clip downstream, with `applied:["upscale:<n>x"]`.
- **Presigned mode** (credentialless: the caller presigns R2): `{ video_url, output_url, output_key, scale?, model? }`.
- **Selftest:** `{ "selftest": true, "scale"? }` generates a real 720p24x3s clip, upscales it end to
  end, and PROVES the result is GPU-bound: it reports the actual `encoder` + `nvenc_used`, the per-phase
  wall-clock split (`phase_s`), sampled GPU/encoder util (`gpu_sample`), `peak_vram_mib`, and
  `batch`/`fp16`. Doubles as the endpoint health check.

`scale` is `2` or `4` (the models are 4x native; a 2x is the 4x inference GPU-downscaled /2, no CPU
Lanczos). `model` is `realesr-animevideov3` (anime/fast, default) or `RealESRGAN_x4plus` (general).
**A non-ok result is a soft-degrade signal** the module honors by passing the original clip through,
never a drop.

## GPU-bound by design (do not regress)

The whole pipeline keeps the GPU fed and the CPU out of the hot path; these are load-bearing:
- **No disk roundtrip; batched fp16.** Frames stream via ffmpeg `rawvideo` pipes; Real-ESRGAN runs on
  BATCHES (`UPSCALE_BATCH`, default 16) in fp16 autocast (weights stay fp32, no `model.half()`). The
  warm-worker model cache is `_MODELS`.
- **GPU rescale + NVENC encode.** Final-size resize is a GPU `interpolate`; the re-encode uses
  `h264_nvenc` when usable. NVENC is probed once per worker (listed AND a real test encode succeeds);
  if not usable it falls back to bounded `libx264` and **reports which encoder ran** (`encoder`), so a
  CPU fallback is never silent.
- **Output cap + wall-clock guards.** Output long edge capped (`MAX_OUTPUT_LONG_EDGE`, default 3840 =
  4K UHD); every ffmpeg phase has a hard `FFMPEG_TIMEOUT` (default 1200s) so a pathological clip
  degrades instead of hanging to the RunPod execution-timeout.

History: `:0.2.5` moved the encode off CPU `libx264` onto `h264_nvenc`; `:0.2.6` made the upscale loop
GPU-bound by removing the per-frame PNG roundtrip and batching in fp16 (issue #7).

## Commands

This is a Python / RunPod image, NOT an npm package. There is no local test suite; verification is the
build-time NVENC assert plus the GPU-gated selftest.

```bash
# Build the image locally (CI does this on push). The build FAILS if h264_nvenc is not compiled into ffmpeg.
docker build -t vivijure-upscale:dev .

# Lint the handler without a GPU.
python -m py_compile handler.py

# GPU verify: send {"selftest": true} on a pinned endpoint / live pod; assert ok:true AND nvenc_used:true.
```

**Tunables (endpoint env):** `MAX_OUTPUT_LONG_EDGE`, `FFMPEG_TIMEOUT`, `UPSCALE_BATCH`, `UPSCALE_TILE`,
`UPSCALE_FP16`, plus the R2 creds (`R2_ENDPOINT_URL` / `R2_BUCKET` / `R2_ACCESS_KEY_ID` /
`R2_SECRET_ACCESS_KEY`). Size `UPSCALE_BATCH` to the card's VRAM (B16 ~8.7 GiB at 720p, ~17 GiB at
1080p). See the README "Tunables" table.

**Release / deploy mechanics.** `.github/workflows/build-image.yml` builds + pushes to GHCR on a push to
`master` (touching the build inputs) as `:latest` + `:<sha>`; a pushed semver tag (`v0.2.6`) ALSO
publishes the bare `:0.2.6` (the immutable tag the endpoint pins to). PUBLIC repo, so CI runs on
GitHub-hosted `ubuntu-latest`. The RunPod endpoint's image tag, **GPU type, and R2 env are dashboard /
endpoint-config knobs** (RunPod's API does not honor them); **container-registry-auth IS now
MCP/API-manageable** (RunPod MCP `create-container-registry-auth` + attach via `containerRegistryAuthId`
on create/update-template, no dashboard step).

## RunPod GPU config

GPU-bound module, so the endpoint should pin a card with hardware NVENC and enough VRAM for the batch,
NOT the cheapest card (GPU-rationing thesis: a faster card finishes in fewer billed seconds). Recommended:
an Ada / Blackwell-Pro card with NVENC (L4 / L40S sm_89, or RTX PRO 6000 sm_120). Avoid cards without
usable NVENC. **No `TORCH_CUDA_ARCH_LIST` to maintain here** (unlike the sibling musetalk image): nothing
compiles from source, torch kernels come from the `runpod/pytorch` cu1281 base, so the image is
GPU-agnostic. GPU type is set on the endpoint, not in this repo.

## Verifying changes

After any handler or Dockerfile change: build clean (the NVENC assert is a build-time fail-fast), then
run `{"selftest": true}` on a real GPU and confirm `ok:true`, `nvenc_used:true`, and a GPU-bound
`gpu_sample` before cutting a release tag. fp16 is effectively lossless here (PSNR ~66 dB vs fp32).

## Source provenance

`handler.py` + `requirements.txt` were RECOVERED verbatim from the published image
`ghcr.io/skyphusion-labs/vivijure-upscale:0.2.2` (the original pod was terminated, never committed); the
Dockerfile is reconstructed from `docker history` (functionally faithful, not byte-identical). The
GPU-bound encode pipeline (NVENC, res cap, GPU rescale, wall-clock guards) was added on top in `:0.2.3`.
Treat the image-extracted files as the source of truth they reconstruct.

## Conventions

- **No em-dashes (U+2014) or en-dashes (U+2013) anywhere.** Use commas, semicolons, parentheses, or `--`.
- Handle / username is `skyphusion` across all services.
- **A CPU fallback is never silent** (report `encoder`); **a degrade is never silent** (the #245 / #249
  discipline): a non-ok result is the module's passthrough signal, never a drop, and never a fake tag.
- Minimal deps; the engine choice (CUDA Real-ESRGAN via spandrel, NOT video2x/Vulkan -- RunPod has no
  working Vulkan stack, proven 2026-06-20) is deliberate. Justify any new dependency.
- Real-ESRGAN (BSD-3-Clause) + spandrel (MIT) + FFmpeg are listed in `THIRD_PARTY_NOTICES.md`; keep it
  current.

## Crew + identity

- The FIRST command in any op is the member's own login shell: `sudo -u <member> bash -lc '<ops>'`
  (loads their `$HOME`, their `~/dev/vivijure-upscale` clone, their gh / RunPod / R2 creds). Commits and
  PRs land under the member's `skyphusion-<member>` identity, never Conrad's.
- Operating memory for the vivijure family lives in the per-project memory under
  `~/.claude/projects/-home-conrad-dev-vivijure/memory/` (`seg-vivijure-modules`); load it before acting.
- **HARD AUP line:** the CSAM bright line is absolute (see the vivijure project memory). Non-negotiable.

## Commits & versioning

Conventional Commits (`feat(scope):`, `fix(scope):`, `docs:`); body explains the why. SemVer-style
`0.MINOR.PATCH` while pre-1.0 (PATCH for fixes / backend tweaks, MINOR for features). A release is a
pushed `vMAJOR.MINOR.PATCH` git tag on `master` (CI publishes the matching immutable image tag).
