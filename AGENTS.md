# Vivijure Upscale

This file is the guide for any coding agent and any contributor. Follow it.

## What this project is

This repo is the GPU backend for the Vivijure `upscale` finish module. One RunPod serverless image upscales video with Real-ESRGAN.

The engine is PyTorch and CUDA through [spandrel](https://github.com/chaiNNer-org/spandrel). Frames stream through ffmpeg pipes. The GPU upscales batches in fp16. The encode uses NVENC. Audio is copied when the clip has audio. There is no per-frame PNG round trip.

The studio module worker lives in `vivijure-cf` and `vivijure-local`. Types come from `vivijure-core`. That worker calls this endpoint.

The image is `ghcr.io/skyphusion-labs/vivijure-upscale`. Pin a versioned tag. See git tags and GHCR. Do not freeze one version here.

The MuseTalk satellite this door used to follow is archived.

The default branch is `main`. Open a pull request. CI must pass.

## Handler contract

`handler.py` takes one typed input and returns one typed output. It has three modes.

R2 finish-chain mode reads and writes the shared bucket. The job carries no storage credentials. Keys: `clip_key`, optional `output_key`, `target_height`, `scale`, and `model`. The result `clip_key` is the new object. `applied` is `upscale:<n>x` or `upscale:<h>h`. The result also reports `out_w` and `out_h`.

Presigned mode uses URLs the caller signed. Keys: `video_url`, `output_url`, `output_key`, and the same optional knobs.

Selftest is `{"selftest": true}`. It builds a 720p clip (24 fps, 3 seconds) and upscales it. It reports `encoder`, `nvenc_used`, `phase_s`, `gpu_sample`, `peak_vram_mib`, `batch`, and `fp16`. It is also the health check. An optional `scale` is allowed.

`target_height` is the studio contract. It is the exact output height. Inference is 4x. The GPU then resizes.

`scale` is `2` or `4` only. The models are 4x native. A 2x request is 4x inference, then a GPU downscale by 2. There is no CPU Lanczos. Any other scale is refused. It is not collapsed.

An unsatisfiable height returns `ok: false`. That includes past 4x, a downscale, a no-op, or a long-edge cap that would change the request.

`model` is `realesr-animevideov3` (default) or `RealESRGAN_x4plus`.

A non-ok result is a soft signal. The module passes the original clip through. It does not drop the shot. It does not invent an `applied` tag.

## GPU path

Keep the CPU off the hot path.

Frames use ffmpeg `rawvideo` pipes. Real-ESRGAN runs in batches (`UPSCALE_BATCH`, default 16) under fp16 autocast. Weights stay fp32. Do not call `model.half()`. The warm cache is `_MODELS`.

The final resize is a GPU interpolate. Encode uses `h264_nvenc` when a real test encode works. If NVENC is not usable, the fallback is bounded `libx264`. The result names `encoder`. A CPU fallback is never silent.

`MAX_OUTPUT_LONG_EDGE` defaults to 3840. `FFMPEG_TIMEOUT` defaults to 1200 seconds. A bad clip degrades. It does not hang until the platform timeout.

History of the encode path is in [CHANGELOG.md](CHANGELOG.md).

## Commands

This is a Python image, not an npm package. CPU tests do not prove the image.

```bash
python -m pytest -q tests/
python -m py_compile handler.py
docker build -t vivijure-upscale:dev .
```

`.github/workflows/ci.yml` runs the first two commands on push and pull request. The image build runs on a `v*.*.*` tag or `workflow_dispatch`. The Docker build fails if ffmpeg has no `h264_nvenc`.

GPU proof sends `{"selftest": true}`. Require `ok: true` and `nvenc_used: true`.

Tunables: `MAX_OUTPUT_LONG_EDGE`, `FFMPEG_TIMEOUT`, `UPSCALE_BATCH`, `UPSCALE_TILE`, `UPSCALE_FP16`, plus the R2 settings. Size `UPSCALE_BATCH` to the card. See the README tunables table.

## Release

`.github/workflows/build-image.yml` publishes to GHCR on a SemVer tag or `workflow_dispatch`. A push to `main` does not publish this image.

A tag publishes the clean version, `major.minor`, a short sha, and `latest`. Do not pin `latest`. The same job publishes the matching `-serve` tags. The tagged commit must be an ancestor of `origin/main`.

Set the endpoint image, the GPU type, and R2 env outside this file. Do not write an endpoint id here.

A version moves only in a release pull request. There is no package manifest. The git tag is the version of record. The tag shape is `v*`.

Before a release pull request, check tags. If you are about to repeat a tag that exists, stop. Read why.

An org admin tags the release pull request merge commit `vX.Y.Z`. Deploy workflows require that commit to be an ancestor of `origin/main`.

The line is 1.x. See [CHANGELOG.md](CHANGELOG.md) and the git tags. Do not freeze a patch number here.

Use a Conventional Commit. The body says why.

## GPU card

Pin a card with hardware NVENC and enough VRAM for the batch. An Ada or Blackwell-Pro card with NVENC fits, such as L4, L40S, or RTX PRO 6000. Avoid a card with no usable NVENC.

This repo has no `TORCH_CUDA_ARCH_LIST`. Kernels come from the `runpod/pytorch` base. The GPU type is an endpoint setting.

## Homelab door

`Dockerfile.serve`, `serve.py`, and `runpod_http_serve.py` add `/run` and `/status` on the same image. The door can stay up on a GPU box you run.

`build-image.yml` publishes `<version>-serve`, `<major>.<minor>-serve`, and `sha-<short>-serve` from the release image that job just built. Use a published tag. A tag you built only on one machine cannot be pulled again by someone else.

`UPSCALE_IMAGE` has no default. Pass it for a hand build. Docker refuses an empty `FROM` at parse time. The `InvalidDefaultArgInFrom` warning is the intended shape.

The default port is 8012 (`PORT`). Bind it to a private address. Do not bind `0.0.0.0`.

`/health` is liveness only. It does not touch the GPU, the model, or the handler. The check that can fail is `POST /run` with `{"selftest": true}`. The overlay forwards that body to the handler.

A default selftest also runs an R2 leg when credentials are present. For a GPU check with no bucket traffic, send `{"selftest": true, "model": "realesr-animevideov3"}`. With a model and without `r2`, the R2 leg is not called.

Measured 2026-08-07 on an RTX 4000 SFF Ada (20475 MiB). Resident memory was 192 MiB before the job, 6392 MiB at peak, and 192 MiB after. `handler.py` calls `torch.cuda.empty_cache()` after the upscale so NVENC can start. The peak follows frame size and batch. Do not quote the peak as a resident footprint. A concurrent peak with a resident speech door was 18797 of 20475 MiB. Check that figure before you share the card.

The speech-door repo `vivijure-audio-upscale` is archived. It used port 8013.

## Verify

After a handler or Dockerfile change, build clean. Then run the GPU proof. Require `ok: true`, `nvenc_used: true`, and a GPU-bound `gpu_sample` before you tag or repin. Verify the artifact. fp16 versus fp32 PSNR is about 66 dB.

A green bake is not a proven upscale. Use a dedicated pod, or a serverless endpoint with min workers at 1. A community pod is not the proof.

If you raise min workers above 0, set them back to 0 when the test ends. Record the before and after values. Do not leave a warm worker because you might test tomorrow.

## Source

`handler.py` and `requirements.txt` were recovered from `ghcr.io/skyphusion-labs/vivijure-upscale:0.2.2`. The Dockerfile was reconstructed from `docker history`. It is functionally faithful. It is not byte-identical. Later tags added the GPU-bound encode path. See [CHANGELOG.md](CHANGELOG.md).

## Docs language and safety

Write Simplified Technical English. Use short sentences. Put one idea in each sentence. Use the active present tense. Use plain words. Use the same word every time.

Do not use an em dash (U+2014). Do not use an en dash (U+2013). Use a comma, a semicolon, parentheses, or two hyphens (`--`).

State the limit next to the capability. Do not turn a measured result into a promise.

Real-ESRGAN (BSD-3-Clause), spandrel (MIT), and FFmpeg are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Update that file when you add a dependency. The engine is CUDA Real-ESRGAN through spandrel. It is not video2x or Vulkan.

The public handle is `skyphusion`.

The CSAM ban includes synthetic images. Do not weaken that check.

Do not put a secret in a tracked file. Do not freeze an endpoint id in this file.

See [README.md](README.md) and [docs/deploy.md](docs/deploy.md).
