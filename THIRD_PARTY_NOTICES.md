# Third-Party Notices -- vivijure-upscale

The wrapper code in this repository (the RunPod handler and Dockerfile) is licensed under
**AGPL-3.0** (see `LICENSE`).

The Docker image this repository builds **incorporates and redistributes** the following
third-party software and pretrained model weights, each under its own license. None carries a
non-commercial restriction.

| Component | Author / Source | License | Notes |
|---|---|---|---|
| Real-ESRGAN models (realesr-animevideov3, RealESRGAN_x4plus) | Xintao Wang et al. -- https://github.com/xinntao/Real-ESRGAN | BSD-3-Clause | Weights baked from upstream public releases. |
| spandrel | chaiNNer-org -- https://github.com/chaiNNer-org/spandrel | MIT | Loads/runs the Real-ESRGAN model under PyTorch. |
| FFmpeg | https://ffmpeg.org | LGPL-2.1 / GPL-2.0 | Frame extract / re-encode; invoked as a subprocess. |
| PyTorch | https://github.com/pytorch/pytorch | BSD-3-Clause | Provided by the base image. |
| Pillow | https://github.com/python-pillow/Pillow | HPND (MIT-style) | Frame I/O. |
| NumPy | https://github.com/numpy/numpy | BSD-3-Clause | Array ops. |

The authoritative copyright line and full license for each component live at its source URL above.
Full license texts: AGPL-3.0 -> `LICENSE`. The MIT and BSD-3-Clause templates that govern the
components above are reproduced below (each component retains its own upstream copyright notice).

---

## MIT License

```
MIT License

Copyright (c) the respective authors of the MIT-licensed components listed above
(chaiNNer-org / spandrel), each retaining its own notice.

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

Copyright (c) 2021, Xintao Wang (Real-ESRGAN); copyright (c) the PyTorch and NumPy
authors for those components -- each retaining its own upstream notice.

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
