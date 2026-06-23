# vivijure-upscale

A RunPod serverless image that upscales video with **Real-ESRGAN**, run through PyTorch/CUDA via
[spandrel](https://github.com/chaiNNer-org/spandrel). The GPU backend for Vivijure's `upscale`
module (#191) -- stream frames through ffmpeg pipes -> upscale in BATCHES on the GPU (fp16) -> re-encode,
audio copied through when present. No per-frame PNG disk roundtrip; the GPU is the bottleneck, not I/O.

## The Vivijure ecosystem

Vivijure is an AI film studio built as a thin control plane plus opt-in GPU modules. These repos
form the constellation; this block is identical in each so the whole map is visible from any one of
them.

```
   friends + Slate (Discord)
            |
            v
        slate  -->  vivijure (studio control plane / JSON API)
                        |
                        v
                  vivijure-backend (GPU render: keyframes -> i2v -> assemble)
                        |
            +-----------+-----------------------------+
            |           |               |             |
            v           v               v             v
     vivijure-     vivijure-       vivijure-      (more finish
     musetalk      upscale         audio-upscale   modules over time)
   (lip-sync)    (video upscale)  (speech enhance)
```

| Repo | Role |
|---|---|
| [slate](https://github.com/skyphusion-labs/slate) | Collaborative AI screenwriter assistant for Discord. Friends and Slate co-author a film in-channel; Slate then submits it to the studio entirely through the vivijure JSON API. |
| [vivijure](https://github.com/skyphusion-labs/vivijure) | The studio control plane (a Cloudflare Worker): planner, cast, and render UI plus the JSON API. A thin module host that orchestrates render jobs behind a typed hook contract. |
| [vivijure-backend](https://github.com/skyphusion-labs/vivijure-backend) | The GPU render backend (RunPod serverless): SDXL keyframes, Wan image-to-video, and ffmpeg assembly. The half that turns a storyboard bundle into a film. |
| [vivijure-musetalk](https://github.com/skyphusion-labs/vivijure-musetalk) | MuseTalk audio-driven lip-sync GPU module (finish-class). Syncs a character's mouth to dialogue audio. |
| [vivijure-upscale](https://github.com/skyphusion-labs/vivijure-upscale) | Real-ESRGAN CUDA video-upscale GPU module (finish-class). Raises the assembled film's resolution. |
| [vivijure-audio-upscale](https://github.com/skyphusion-labs/vivijure-audio-upscale) | CUDA speech-audio enhancement (resemble-enhance) GPU module. The GPU half of the cost-aware audio finish path. |

## Team

Vivijure is built by Conrad (`skyphusion`) and his named AI crew. The crew are treated as
individuals, each working in their own lane with their own GitHub identity; this is the same
transparent framing used across the project.

| Member | Role | GitHub |
|---|---|---|
| Conrad | Creator / director | [@skyphusion](https://github.com/skyphusion) |
| Mackaye | PM / tech lead | [@skyphusion-mackaye](https://github.com/skyphusion-mackaye) |
| Strummer | Infrastructure | [@skyphusion-strummer](https://github.com/skyphusion-strummer) |
| Rollins | Backend / modules | [@skyphusion-rollins](https://github.com/skyphusion-rollins) |
| Joan | Frontend / extraction | [@skyphusion-joan](https://github.com/skyphusion-joan) |

## GPU-bound by design

The whole pipeline keeps the GPU busy and the CPU out of the hot path:

- **No disk roundtrip; batched fp16 inference.** Frames stream in/out via ffmpeg `rawvideo` pipes (no
  per-frame PNG read/write), and Real-ESRGAN runs on BATCHES of frames (`UPSCALE_BATCH`) in fp16
  (autocast). This is what keeps the GPU fed: the upscale phase went from ~25s at ~12% GPU util to
  ~3.5s at 100% peak util on an RTX 6000 Ada (720p24x3s, 2x), a ~7x speedup, output unchanged.
- **Inference + rescale on the GPU.** Real-ESRGAN runs on CUDA; the resize to the final frame size
  (for a 2x request, and/or the resolution cap) is a GPU `interpolate`, not a CPU Lanczos pass.
- **NVENC re-encode.** The final encode uses `h264_nvenc` (hardware encode) when the card + ffmpeg
  support it. The encoder is probed once per worker (listed **and** a real test encode succeeds); if
  NVENC is not usable, it falls back to a bounded `libx264` and **reports which encoder ran** in the
  result (`encoder`), so a CPU fallback is never silent.
- **Output resolution cap.** The models are 4x native, so a 4x of 1080p would be 8K (7680x4320). The
  output long edge is capped (default **2160p / 4K UHD**, `MAX_OUTPUT_LONG_EDGE`) and a 2x request is
  produced at 2x, so the encode and the in-memory frame buffers stay bounded regardless of source.
- **Wall-clock guards.** Every ffmpeg phase has a hard timeout (`FFMPEG_TIMEOUT`, default 1200s). A
  pathological clip degrades (ok:false -> the module passes the original through) instead of hanging
  to the RunPod execution-timeout.

History: `:0.2.5` moved the re-encode off CPU `libx264` onto `h264_nvenc` (the original ship-blocker --
the encode pegged the CPU for minutes). `:0.2.6` then made the upscale loop itself GPU-bound by removing
the per-frame PNG disk roundtrip and batching the inference in fp16 (issue #7).

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

The models are 4x native; a requested `scale` of 2 is produced at 2x by a GPU downscale of the 4x
inference output (no CPU Lanczos pass, no oversized intermediate on disk).

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

Self-test (no R2 -- confirms CUDA, loads the model, upscales a generated 720p24 x 3s clip end to end,
and **proves the result is GPU-bound**):

```json
{ "selftest": true, "scale": 2 }
```

The self-test result reports `encoder` + `nvenc_used` (which encoder actually ran), `batch` + `fp16`
(the inference settings used), `phase_s` (the decode / upscale / encode wall-clock split), `gpu_sample`
(sampled GPU + NVENC utilization, max/avg), `peak_vram_mib`, and `input_res` / `output_res`.

A non-ok result is a soft-degrade signal to the caller (pass the original through), never a drop.

## Tunables (endpoint env)

| Env | Default | Effect |
|-----|---------|--------|
| `MAX_OUTPUT_LONG_EDGE` | `3840` | Output long-edge cap in px (2160p / 4K UHD). Bounds worst-case wall-clock. |
| `FFMPEG_TIMEOUT` | `1200` | Per-phase wall-clock guard (s). Exceeding it degrades the job, never hangs. |
| `UPSCALE_BATCH` | `16` | Frames per GPU inference batch. Higher = better GPU saturation, more VRAM (B16/720p ~8.7 GiB, /1080p ~17 GiB). Lower it on a smaller card. |
| `UPSCALE_TILE` | `1024` | Tile size (px) for the tiled inference. Larger = fewer kernel launches (higher util), more VRAM. |
| `UPSCALE_FP16` | `1` | fp16 inference via autocast (set `0` for fp32). fp16 is effectively lossless here (PSNR ~66 dB vs fp32, max 1 LSB). |
| `R2_ENDPOINT_URL` / `R2_BUCKET` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` | -- | R2 mode credentials (set in the RunPod dashboard). |

## RunPod GPU config (which card to pin the endpoint to)

This is a **GPU-bound** module (batched fp16 Real-ESRGAN + NVENC), so the endpoint should be pinned
to a card with hardware NVENC and enough VRAM for the batch -- not the cheapest card. Per the
GPU-rationing thesis, a faster card finishes the job in fewer billed seconds, so speed-per-dollar
wins over sticker price.

- **Recommended:** an Ada / Blackwell-Pro card with NVENC -- e.g. **L4 / L40S (Ada, sm_89)** or
  **RTX PRO 6000 (Blackwell, sm_120)**. The reference number in "GPU-bound by design" (~3.5s for a
  720p24x3s 2x upscale at 100% util) is on an **RTX 6000 Ada**.
- **VRAM vs batch:** size `UPSCALE_BATCH` to the card. B16 needs ~8.7 GiB at 720p, ~17 GiB at 1080p;
  a 24 GiB card runs B16/1080p comfortably, a 16 GiB card should drop the batch. `UPSCALE_TILE`
  trades VRAM for kernel-launch overhead the same way.
- **Avoid:** cards without usable NVENC -- the encode then falls back to bounded `libx264` on CPU
  (the result reports `encoder` so the fallback is never silent, but it is slower and off-thesis).
- **Where it is set:** the GPU type is selected on the **RunPod endpoint** (dashboard / endpoint
  config), not in this repo. The image is GPU-agnostic; torch kernels come from the
  `runpod/pytorch` cu1281 base, so there is **no `TORCH_CUDA_ARCH_LIST` to maintain here** (unlike
  the sibling musetalk image, which compiles mmcv CUDA ops and does pin an arch list). Endpoint env
  + GPU + registry-auth are the deliberate, dashboard-set knobs (RunPod's API does not honor them).

## Source provenance

This repository's source was **recovered from the published image**
`ghcr.io/skyphusion-labs/vivijure-upscale:0.2.2`: the original was built on a since-terminated RunPod
pod and was never committed, so the image was the only surviving copy. `handler.py` and
`requirements.txt` are verbatim from the image; the `Dockerfile` is reconstructed from
`docker history` (functionally faithful, not byte-identical). The GPU-bound encode pipeline above
(NVENC, resolution cap, GPU rescale, wall-clock guards) was added on top in `:0.2.3`.

## License

**AGPL-3.0-only.** A labor of love, given freely: use it, learn from it, self-host it, build your own creative visions on it. Run it as a network service and the AGPL has you share your changes back, so it stays a commons. It is not for sale, and not to be resold as a SaaS.

Third-party components it incorporates (Real-ESRGAN -- BSD-3-Clause; spandrel -- MIT; FFmpeg) are listed in `THIRD_PARTY_NOTICES.md`.
