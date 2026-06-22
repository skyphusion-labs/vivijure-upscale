# vivijure-upscale

A RunPod serverless image that upscales video with [video2x](https://github.com/k4yt3x/video2x)
(Real-ESRGAN / RealCUGAN / Anime4K). The GPU backend for Vivijure's `upscale` module (#191) -- the
"professional-grade, given away" brick: the single biggest perceived-quality jump in the stack.

## Why video2x (not Real-ESRGAN direct)
video2x wraps the upscalers AND owns the full video pipeline (extract -> upscale -> re-encode). On
maintenance it's the better-kept path: live source (commits into 2026) vs the original Real-ESRGAN
repo (untouched 2+ years). Its upstream Docker image is stale, so **we build from current source**.

## Footprint (why it's cheap -- and a separate endpoint)
This is the lightweight opposite of `vivijure-backend`, so it runs as its OWN serverless endpoint:
- **No network volume.** The Real-ESRGAN models are a few MB; they are **baked into the image** (NOT
  on an attached volume). That removes the model-seeding step entirely -- nothing to provision/attach.
- **Small GPU, low power.** v6 is Vulkan + ncnn -- **no PyTorch, no CUDA toolkit** for inference. Runs
  on a modest GPU (e.g. an L4 / A4000-class) with a fast cold start and cheap per-second cost. Don't
  put this on the B200/H200 the backend uses; size it down.
- Implication: simplest possible RunPod endpoint -- image only, no volume, small GPU, scale-to-zero.

## License boundary
video2x is **AGPL-3.0**. The handler runs it as a SEPARATE PROCESS (`subprocess` to the `video2x` CLI),
so Vivijure's Worker/module code never links it (mere aggregation). This image distributes the GPL
binary; its source is upstream at k4yt3x/video2x.

## Handler contract (job input)
Presigned-URL transport, identical in spirit to `/film-titles` and the i2v modules -- the core holds
the R2 creds and presigns; the handler is credentialless.
```json
{
  "video_url":  "<presigned R2 GET of the source clip>",
  "output_url": "<presigned R2 PUT for the result>",
  "output_key": "renders/<project>/clips/<shot>_up.mp4",
  "scale": 2,
  "processor": "realesrgan",
  "model": "realesr-animevideov3"
}
```
Returns `{ ok, output_key, bytes, scale, processor }`, or `{ ok:false, error }` (the module
soft-degrades to passthrough on a non-ok result -- never drops a clip).

## Build + deploy
```bash
docker build -t ghcr.io/skyphusion-labs/vivijure-upscale:0.1.0 .
docker push  ghcr.io/skyphusion-labs/vivijure-upscale:0.1.0
# then: RunPod template -> serverless endpoint (GPU type with Vulkan; A-series / L40 / etc.)
```
Same release shape as vivijure-backend (GHCR + Jenkins tag-build); wire it into the `upscale` module
as `RUNPOD_ENDPOINT` once the endpoint id exists.

## RESOLVED (verified against k4yt3x/video2x master + release 6.4.0, 2026-06-20)
- **`just ubuntu2204` recipe** -- confirmed valid (build file is `.justfile`, default branch `master`).
  It adds `ppa:ubuntuhandbook1/ffmpeg7` and builds against FFmpeg 7.
- **FFmpeg 7 runtime libs** -- the build links FFmpeg 7, so the runtime stage now adds the SAME PPA
  (stock 22.04 ffmpeg 4.x would fail the .deb's dynamic links). Fixed in the Dockerfile runtime stage.
- **Model files** -- ship IN the repo (`models/{realesrgan,realcugan,libplacebo,rife}`) and the CMake
  rule `install(DIRECTORY models -> share/video2x)` packages them INTO the `.deb`. No manual baking;
  installing the .deb gives `/usr/share/video2x/models`.

## OPEN ITEMS (verify on a real GPU pod -- the fiddly Vulkan bits)
1. **Vulkan ICD on RunPod**: v6 needs the NVIDIA Vulkan ICD. It should be injected by the nvidia
   container runtime; verify `vulkaninfo` sees the GPU inside the running pod. If not, mount/install
   the ICD JSON (`/usr/share/vulkan/icd.d/nvidia_icd.json`).
2. **GPU selection**: video2x picks GPU via `-g`; default 0 should be fine on a single-GPU pod.
5. **Audio/format**: confirm video2x preserves the clip's audio (likely re-encodes video only). If it
   drops audio, the module/handler must remux it back (clips are usually silent pre-score, so low risk).
6. **Encoder**: default codec; set `-c libx264` + a sane CRF for web-playable output if needed.
