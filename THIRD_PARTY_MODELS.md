# Third-party models (Hub / baked image)

The `ghcr.io/skyphusion-labs/vivijure-upscale` image bakes Real-ESRGAN weights so a worker runs
without a network pull. This is the Hub-facing summary. Full copyright and license text:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

| Role | Component | License | Source |
| --- | --- | --- | --- |
| Upscale models | realesr-animevideov3, RealESRGAN_x4plus | BSD-3-Clause | https://github.com/xinntao/Real-ESRGAN |
| Model runner | spandrel | MIT | https://github.com/chaiNNer-org/spandrel |
| Encode / decode | FFmpeg | LGPL-2.1 / GPL-2.0 | https://ffmpeg.org |
| Runtime | PyTorch | BSD-3-Clause | https://github.com/pytorch/pytorch |

Wrapper code in this repository is **AGPL-3.0** (see `LICENSE`). None of the baked models carries a
non-commercial restriction.
