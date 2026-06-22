# Third-Party Notices -- vivijure-upscale

The wrapper code in this repository (the RunPod handler and Dockerfile) is licensed
under **AGPL-3.0** (see `LICENSE`).

The Docker image this repository builds **incorporates and redistributes** the
following third-party software and pretrained model weights, each under its own
license. The video upscaler **video2x** is invoked as a separate process
(`subprocess` to the `video2x` CLI), so the wrapper code is not linked against it
(mere aggregation); the image nonetheless distributes the video2x binary, so its
license and a source offer are reproduced below. None of these carries a
non-commercial restriction.

| Component | Author / Source | License | Notes |
|---|---|---|---|
| video2x | k4yt3x -- https://github.com/k4yt3x/video2x | **AGPL-3.0** | Distributed in the image; driven as a subprocess. |
| Real-ESRGAN models (realesr-animevideov3, RealESRGAN_x4plus) | Xintao Wang et al. -- https://github.com/xinntao/Real-ESRGAN | BSD-3-Clause | Weights baked from upstream releases. |
| Anime4K (bundled in video2x) | bloc97 -- https://github.com/bloc97/Anime4K | MIT | Ships inside the video2x binary. |
| Real-CUGAN ncnn (bundled in video2x) | nihui -- https://github.com/nihui/realcugan-ncnn-vulkan | MIT | Ships inside the video2x binary. |
| RIFE ncnn (bundled in video2x) | nihui -- https://github.com/nihui/rife-ncnn-vulkan | MIT | Ships inside the video2x binary. |
| ncnn | Tencent -- https://github.com/Tencent/ncnn | BSD-3-Clause | Inference runtime bundled in video2x. |
| FFmpeg | https://ffmpeg.org | LGPL-2.1 / GPL-2.0 | Bundled in video2x; see upstream for build config. |

### video2x (AGPL-3.0) -- source offer

This image distributes the **video2x** binary, licensed under the GNU Affero
General Public License v3.0. The complete corresponding source code for video2x
is available from its upstream repository: **https://github.com/k4yt3x/video2x**.
video2x is run as an unmodified separate process; this repository does not modify
it. The full AGPL-3.0 text is in this repository's `LICENSE`.

The full license texts for the bundled upscalers (Anime4K, Real-CUGAN, RIFE-ncnn)
and ncnn are documented in video2x's own NOTICE file at the source URL above. The
MIT and BSD-3-Clause templates that govern Real-ESRGAN, the bundled upscalers, and
ncnn are reproduced below (each component retains its own upstream copyright).

---

## MIT License

```
MIT License

Copyright (c) the respective authors of the MIT-licensed components listed above
(bloc97 / nihui), each retaining its own notice.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## BSD 3-Clause License

```
BSD 3-Clause License

Copyright (c) 2021, Xintao Wang (Real-ESRGAN) and 2017, Tencent (ncnn), all
rights reserved, each retaining its own upstream notice.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
