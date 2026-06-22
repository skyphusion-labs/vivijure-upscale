# vivijure-upscale

A RunPod serverless image that upscales video with **Real-ESRGAN**, run through PyTorch/CUDA via
[spandrel](https://github.com/chaiNNer-org/spandrel). The GPU backend for Vivijure's `upscale`
module (#191) -- extract frames -> upscale each (tiled to bound GPU memory) -> re-encode, audio
copied through when present.

## Engine: CUDA Real-ESRGAN (not Vulkan/video2x)

An earlier attempt wrapped [video2x](https://github.com/k4yt3x/video2x) (Vulkan). RunPod has no
working Vulkan stack (proven 2026-06-20), so the engine was swapped to **Real-ESRGAN on PyTorch/CUDA
via spandrel** -- the same Real-ESRGAN models, a CUDA engine instead of Vulkan. The transport
contract and the `{"selftest": true}` harness are unchanged from that attempt.

## Models

The Real-ESRGAN weights are a few MB and are **baked into the image** (no network volume), pulled
from xinntao's public releases:

- `realesr-animevideov3` -- anime / fast (default).
- `RealESRGAN_x4plus` -- general-purpose 4x.

The models are 4x; a requested `scale` of 2 upscales 4x then downscales to 2x with a Lanczos pass.

## Job input

R2 mode (the finish-chain module contract -- the endpoint reads/writes the shared bucket itself):

```json
{ "clip_key": "renders/<project>/clips/<shot>_i2v.mp4", "output_key": "...optional...",
  "scale": 2, "model": "realesr-animevideov3" }
```

Presigned mode (credentialless -- the caller presigns R2):

```json
{ "video_url": "<presigned GET>", "output_url": "<presigned PUT>",
  "output_key": "renders/<project>/clips/<shot>_up.mp4", "scale": 2, "model": "realesr-animevideov3" }
```

Self-test (no R2 -- confirms CUDA, loads the model, upscales a generated clip end to end):

```json
{ "selftest": true }
```

A non-ok result is a soft-degrade signal to the caller (pass the original through), never a drop.

## Source provenance

This repository's source was **recovered from the published image**
`ghcr.io/skyphusion-labs/vivijure-upscale:0.2.2`: the original was built on a since-terminated RunPod
pod and was never committed, so the image was the only surviving copy. `handler.py` and
`requirements.txt` are verbatim from the image; the `Dockerfile` is reconstructed from
`docker history` (functionally faithful, not byte-identical).

## License

This wrapper is licensed under **AGPL-3.0** (see `LICENSE`). Third-party components it incorporates
(Real-ESRGAN -- BSD-3-Clause; spandrel -- MIT; FFmpeg) are listed in `THIRD_PARTY_NOTICES.md`.
