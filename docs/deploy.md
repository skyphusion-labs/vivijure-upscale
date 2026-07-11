# Deploy the video upscale finish engine

This page walks you through standing up `vivijure-upscale` on your own. When you finish, you will
have a RunPod endpoint that makes finished video sharper and higher-resolution (Real-ESRGAN), and an
endpoint id you paste into your Vivijure Studio to turn it on.

New here? The one-page picture of how the parts fit together is in
[constellation.md](constellation.md). This engine is one box on that map.

## What you need first

- A **RunPod** account, and an **API key** from it (runpod.io, then Settings, then API Keys).
  RunPod is where the GPU runs.
- **Docker** on your computer, so you can build the image.
- A **registry** to push the image to, and to be logged in to it (for example GitHub Container
  Registry, `ghcr.io`).
- Optional, for the studio's normal mode: **R2 storage keys** (Cloudflare R2). The endpoint reads the
  clip from R2 and writes the upscaled result back to R2.

## The short path

```bash
cp deploy.env.example deploy.env   # then open deploy.env and fill in your keys
./deploy.sh                        # safe to re-run
```

The script builds the image, pushes it, creates the RunPod endpoint, and prints the endpoint id. It
stops on the first error, so you never end up half-deployed.

## What the script does, step by step

1. **Builds** the Docker image from this repo.
2. **Pushes** it to your registry.
3. **Creates a RunPod template and endpoint** (or reuses them if they already exist), pinned to the
   GPU you chose, set to scale to zero.
4. **Prints the endpoint id** and reminds you how to wire it into the studio.

## Every setting you can set

All settings live in `deploy.env`. The example file has them all with comments; here is what each one
means and why.

### The keys you must set

- **`RUNPOD_API_KEY`** -- your RunPod API key. Why: the script talks to RunPod for you to make the
  endpoint. Example: `RUNPOD_API_KEY=rpa_XXXX...`.
- **`IMAGE`** -- the image name to build and run. Why: it is both where the script pushes the image
  and what the endpoint pulls. Point it at your own registry. Example:
  `IMAGE=ghcr.io/yourname/vivijure-upscale:latest`.
- **`ENDPOINT_NAME`** -- a label for the endpoint. Why: the script finds and reuses an endpoint by
  this name, so re-running is safe. Example: `ENDPOINT_NAME=vivijure-video-upscale`.
- **`GPU_TYPE_IDS`** -- which GPU cards RunPod may use, separated by commas. Why: this job needs
  **hardware video encode (NVENC)** and only a few GB of VRAM, so it does not need a giant card. An
  **Ada** card like the **L4** or **L40S** is the sweet spot; a **Blackwell RTX PRO 6000** also works
  if Ada stock is thin. A faster card finishes in fewer billed seconds, so speed per dollar beats
  sticker price. Example: `GPU_TYPE_IDS=NVIDIA L4,NVIDIA L40S`.

### The knobs you usually leave alone

- **`CONTAINER_DISK_GB`** (default `20`) -- how much disk the container gets. Why: the Real-ESRGAN
  weights are only a few MB, so 20 is plenty. Example: `CONTAINER_DISK_GB=20`.
- **`WORKERS_MIN`** (default `0`) -- the fewest workers kept running. Why: `0` means scale to zero, so
  you pay nothing when no one is rendering.
- **`WORKERS_MAX`** (default `2`) -- the most workers that can run at once. Why: caps parallel jobs and
  your spend.
- **`IDLE_TIMEOUT`** (default `5`) -- seconds a worker stays warm after a job before it shuts down.
  Why: a small warm window avoids a cold start if a second clip arrives right away.
- **`EXECUTION_TIMEOUT_MS`** (default `600000`) -- the longest a single job may run, in milliseconds
  (600000 = 10 minutes). Why: a stuck job is cut off instead of billing forever.
- **`CONTAINER_REGISTRY_AUTH_ID`** (default empty) -- a RunPod credential id for a **private** image.
  Why: if your image is private, RunPod needs a login to pull it. Make one in the RunPod console
  (Settings, then Container Registry Auth) and paste its id here. Leave blank for a public image.
- **`REGISTRY_USER`** / **`REGISTRY_TOKEN`** (default empty) -- a login for your registry, used to push
  the image. Leave blank if you already ran `docker login`.
- **`SKIP_BUILD`** (default `0`) -- set `1` to skip build and push and reuse an image already pushed.
  **`SKIP_ENDPOINT`** (default `0`) -- set `1` to stop after pushing the image (no endpoint).
- **`VERIFY`** (default `0`) -- set `1` to run a post-deploy health check: the script submits the sweep
  self-test to the live endpoint and FAILS CLOSED unless it comes back ok. Why: it proves the freshly
  deployed image actually upscales on the card (every shipped model, plus the R2 finish round-trip when R2
  creds are set) before you repin. It spends a little GPU time, so it is off by default. `VERIFY_TIMEOUT`
  (default `900`) caps how long it polls, in seconds. Example: `VERIFY=1`.

### The upscaler's own tuning knobs

These control how the upscaler runs. They are set on the endpoint (the script passes them through if
you set them). The defaults are good for most cards.

- **`MAX_OUTPUT_LONG_EDGE`** (default `3840`) -- the biggest the long side of the output may be, in
  pixels (3840 = 4K UHD). Why: the model is 4x native, so a 4x of 1080p would be 8K; this cap keeps the
  encode and memory sane no matter the source. Example: `MAX_OUTPUT_LONG_EDGE=3840`.
- **`FFMPEG_TIMEOUT`** (default `1200`) -- a per-step wall-clock guard, in seconds. Why: a pathological
  clip degrades (passes the original through) instead of hanging. Example: `FFMPEG_TIMEOUT=1200`.
- **`UPSCALE_BATCH`** (default `16`) -- how many frames the GPU upscales at once. Why: bigger batches
  keep the GPU busier but use more VRAM. Lower it on a smaller card. On a CUDA out-of-memory the handler
  automatically splits the batch (halving down to a single frame, freeing the cache between tries) and
  retries, so a heavy model can never hard-OOM -- worst case it runs one frame per tile (slow, correct).
  Example: `UPSCALE_BATCH=8` on a 16GB card.
- **`UPSCALE_TILE`** (default `512`) -- the tile size, in pixels, the model works in. Why: larger
  tiles mean fewer GPU launches (faster) but more VRAM. It MUST be smaller than the frame or tiling is a
  no-op and the whole frame runs in one forward -- that is how a native-4x model (RealESRGAN_x4plus, a
  heavy RRDB) ran a full-frame 16-frame batch and hit CUDA OOM. 512 genuinely subdivides a 720p frame.
  **Card sizing:** `RealESRGAN_x4plus` at the default `512` tile wants a large card (>= ~80 GB); on a
  smaller card (~48 GB class) set `UPSCALE_TILE=256`. If a single frame still will not fit, the handler
  now auto-shrinks the tile too (see `UPSCALE_TILE_FLOOR`). The fast `realesr-animevideov3` model is fine
  on small cards at the default. Example: `UPSCALE_TILE=512` (or `256` on a smaller card).
- **`UPSCALE_FP16`** (default `1`) -- run in half precision (`1`) or full precision (`0`). Why: half
  precision is about twice as memory-friendly and effectively lossless here (about 66 dB versus full,
  which is imperceptible). Example: `UPSCALE_FP16=1`.
- **`UPSCALE_TILE_FLOOR`** (default `64`) -- the smallest tile the auto-shrink fallback will drop to, in
  pixels. Why: when the batch split has already reached a single frame and that frame STILL will not fit
  (a card too small for one frame at `UPSCALE_TILE` through a heavy 4x model), the handler halves the tile
  and retries, down to this floor, so the frame still upscales (slower, correct) instead of hard-failing.
  Raise it if the smallest tiles are too slow; lower it only on a very tight card. Example:
  `UPSCALE_TILE_FLOOR=64`.
- **`PYTORCH_CUDA_ALLOC_CONF`** (default `expandable_segments:True`) -- how PyTorch grows its CUDA memory
  pool. Why: `expandable_segments:True` lets the allocator grow and release segments instead of stranding
  reserved-but-unused VRAM as fragmentation, which is what filled the card at the edge of the x4plus fit;
  it drops the total footprint well under the raw allocation peak. The handler sets this before it loads
  PyTorch; set your own value here to override it. Example: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### The endpoint's own settings (R2 mode)

The studio's normal mode is "finish-chain" mode: the endpoint reads and writes your R2 bucket by key,
so no clip data passes through the studio. Set these four to turn it on. Leave them blank to use only
the presigned-URL mode, where the studio hands the endpoint short-lived links instead.

- **`R2_ENDPOINT_URL`** -- your R2 S3 address (looks like `https://<account>.r2.cloudflarestorage.com`).
- **`R2_BUCKET`** (default `vivijure`) -- the bucket name the clips live in.
- **`R2_ACCESS_KEY_ID`** / **`R2_SECRET_ACCESS_KEY`** -- an R2 key pair scoped to that bucket. Make a
  key just for this engine so its reach is small.

## What the endpoint expects as a job

You do not call this by hand in normal use; the studio does. But so you know exactly what it does,
here is the contract.

- **R2 finish-chain mode:** `{ "clip_key": "...", "output_key": "...", "scale": 2,
  "model": "realesr-animevideov3" }`. The endpoint reads the clip from R2 and writes the result to
  `output_key`.
- **Presigned mode:** `{ "video_url": "...", "output_url": "...", "output_key": "...", "scale": 2 }`.
  The studio presigns the links; the endpoint holds no keys.
- **Self-test:** `{ "selftest": true, "scale": 2 }`. Upscales a generated clip end to end and reports
  which encoder ran, the GPU use, and the timing, so you can prove a fresh endpoint is GPU-bound. With no
  `model` it SWEEPS every shipped model on the real GPU (so a heavy model like `RealESRGAN_x4plus` is
  verified, not just the default) AND runs the R2 finish-contract round-trip (the real bucket
  download+upload path, #26). The R2 leg is opportunistic: it HONEST-SKIPS when the endpoint has no R2
  creds (reported, never a silent pass); pass `"r2": true` to REQUIRE it (absent creds then fail). `ok`
  is true only when every model passed and the R2 leg did not fail. Run this before you repin (or just set
  `VERIFY=1` on the deploy).

Two job knobs you can pass:

- **`scale`** (default `2`) -- how much bigger to make the video, `2` or `4`. Why: `2` doubles the
  resolution, `4` quadruples it. Anything `4` or higher is treated as `4`. Example: `scale: 2`.
- **`model`** (default `realesr-animevideov3`) -- which Real-ESRGAN model to use. Why:
  `realesr-animevideov3` is fast and great for animation; `RealESRGAN_x4plus` is a general-purpose 4x
  model. Example: `model: "realesr-animevideov3"`.

The result reports the encoder that ran, so a CPU fallback is never silent. If a clip cannot be
processed, the engine signals a soft-degrade (pass the original through) instead of dropping it.

## Turn it on in the studio

This engine powers the studio's **finish-upscale** module (an opt-in tier). To turn it on:

1. Copy the endpoint id the script printed.
2. In your studio's `deploy.env`, set **`VIDEO_UPSCALE_RUNPOD_ENDPOINT_ID`** to that id.
3. Keep `VIVIJURE_PROFILE=full` and re-run the studio's `./deploy.sh`.

Full context on the tiers is in the studio's `docs/opt-in-tiers.md` (the "finish-upscale" entry).

## Re-running and fixing things

- Re-running `./deploy.sh` is safe. It reuses the template and endpoint it already made.
- To change the endpoint's GPU or scaling after it exists, use the RunPod console; RunPod does not let
  the API re-pin an endpoint's GPU list after creation.
- If a push fails, make sure you ran `docker login` for your registry and that the repo exists there.
- If the encode is slow and pegs the CPU, your card has no usable NVENC; the self-test result names the
  encoder that ran, so pick a card with hardware NVENC (L4 / L40S / Blackwell).
