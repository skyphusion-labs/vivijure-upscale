# Changelog -- vivijure-upscale

The ESRGAN video-upscale finish satellite. The image ships as a git-tag-driven release
(`v<X.Y.Z>`; CI publishes the GHCR consumer image on tag push). The tag is the version of record,
and a tag publishing an image is **not** the same act as production being pinned to it: repins are
manual and spend-gated, and several tags below deliberately ship an image nothing repins to. This
file records the why behind each release. Newest first.

## Unreleased

- **fix(serve): an oversize or unparseable POST /run is no longer accepted as an empty job (#106).**
  `_body()` answered `None` for no body, a body past the 1 MiB cap, and a body that would not
  parse, and `/run` then did `(body or {}).get("input", body or {})`, so all three were accepted
  with `200` and a job id. The caller got a success shape for a request that was never honoured,
  and the job failed later naming a missing field rather than the body. Now `413` and `400`
  respectively, checked AFTER authentication so an unauthenticated caller still gets `401` and
  learns nothing about the cap. Ported from vivijure-blender.

- **fix(security): SSRF-gate presigned GET/PUT and redact query strings in errors.** Presigned
  `video_url` / `output_url` / `hash_url` now follow the musetalk / audio-upscale guard: https
  only, no localhost, DNS-resolve and refuse private / loopback / link-local / metadata
  addresses, optional `R2_URL_HOST_SUFFIX` pin when set, and connect to the validated IP with
  `allow_redirects=False` so a 30x cannot bounce onto another host. Exception text returned
  to the caller has query strings stripped (`?[redacted]`) so an X-Amz presign never leaves
  the worker. Guarded by `tests/test_url_validation.py`.

## v1.1.4 -- 2026-08-16

A long `RealESRGAN_x4plus` shot no longer burns the whole guard and ships the
source as a successful film. Pin serve to `1.1.4-serve` after the bake if you
want this on the twins; 1.1.3 still honours `target_height` and is otherwise
the live pin.

- **fix(upscale): refuse a job that cannot finish inside the budget, instead of burning the
  guard and shipping the source (#98).** #99 released the abort-path leak and stopped
  re-searching the tile; #105 made a timeout return `{ok: false, detail}` so the module
  degrades. What stayed: `RealESRGAN_x4plus` on a shot past ~10s of 720p still spent the
  whole 1200s (serve) / 570s (serverless) guard, then the module passthrough'd and the
  film shipped un-upscaled. `SCENE_MAX_SECONDS` is 60. Raising the guard just meets
  `PHASE_HARD_DEADLINE_SECONDS` (5400) instead.

  After the first batch that did not pay an OOM-search, the loop now projects
  `settled_seconds_per_frame * frames_left` against the remaining budget and raises
  `InvocationExpired` immediately. Same envelope the module already treats as degrade.
  A first batch that shrank (x4plus 512 -> 256) is not the rate -- projecting from it
  would refuse shots the remaining batches would have finished. Never `ok: true` at the
  source resolution; never an upload of a partial artifact.

## v1.1.3 -- 2026-08-16

The live serve pin (`1.1.1-serve`) still ignored `target_height` and
collapsed every scale to 2 or 4. This tag is the first image that honours
the studio height contract (#109) and that caps the encode inside the
invocation deadline (#107). Pin serve to `1.1.3-serve` after the bake;
do not leave twins on 1.1.1.

- **fix(upscale): honour `target_height` and stop collapsing `scale` to 2 or 4 (#102).** The studio
  (vivijure-local `local-finish/app.ts`) sends `target_height: 1080` and no scale. The handler never
  read that key. Sizing was one line -- `final_scale = 4 if int(inp.get("scale", 2) or 2) >= 4 else 2`
  -- so a sized request returned 2x the SOURCE with `ok: true`. Hosted upscale is a billed GPU door;
  that is silent wrong output we still charge for. It only produced 1080 when the source was exactly
  540p, the one fixture where honoured and ignored are indistinguishable.

  One function (`_resolve_output_size`) is now the sizing authority. `target_height` delivers that
  exact even height (aspect-preserved) via the GPU interpolate the pipeline already had after native
  4x. `scale` is only 2 or 4; any other value is refused, never collapsed. A height that needs more
  than 4x, a downscale, a no-op (already at the target), or a long-edge cap that would change a
  requested height returns `ok: false` so the module passthroughs instead of substituting. A
  scale-only request may still be long-edge-capped (the safety bound stays), but the result reports
  the actual `out_w` / `out_h` / `scale` and the applied tag is `upscale:<h>h` unless the requested
  2x/4x was delivered exactly.

- **fix(upscale): bring the ENCODE step inside the invocation deadline, and enforce it ON the child
  processes (#105).** `_upscale_video` stamped one deadline and checked it twice, at decode and at
  upscale. Between `t2` and `t3` -- a raw `subprocess.Popen`, a write loop, and an unbounded
  `enc.wait()` -- there was no check and no timeout, so the invocation was not capped by anything. It
  read as complete because the module carried `_run(cmd, timeout=FFMPEG_TIMEOUT)`, which looked like
  the convention the encode path had opted out of. **Measured at `d34135d`: that helper had ZERO
  callers in the entire repo.** The convention it advertised was honoured by none of the module's
  subprocess sites, not one; the encode path was not an exception to it, there was no rule. It is
  replaced by `_run_bounded(cmd, timeout, ...)` with no default for `timeout`, and every
  `subprocess.run` in the file now goes through it.

  Coverage is now **19 of 19 subprocess and network sites bounded** across `handler.py`,
  `runpod_http_serve.py` and `serve.py` (the last two hold none), against 7 of 19 before. The two
  `Popen` children are additionally watchdogged: a deadline check between loop iterations cannot
  interrupt a write to a pipe nothing is draining or a `wait()` on a stalled child, because both block
  in the kernel where the next check is never reached. Killing the child is what unblocks them, so the
  guard acts ON the process rather than only observing the clock. `_load_model` is deliberately
  outside the budget (a local weights read on a warm-cached path, before the deadline is stamped).

- **fix(upscale): size the guard UNDER the platform ceiling instead of at twice it (#105).** A guard
  above the platform's own execution timeout can never fire: the platform gets there first, and its
  kill leaves a job-level FAILED envelope with no structured output, which the studio must treat as a
  crash and which fails the whole film. `deploy.sh` has passed `executionTimeoutMs=600000` (600s) on
  every endpoint it creates since 2026-07-01, while `FFMPEG_TIMEOUT` defaults to 1200. The budget is
  now `min(FFMPEG_TIMEOUT, UPSCALE_PLATFORM_TIMEOUT - UPSCALE_PLATFORM_MARGIN)`, which is **570s** with
  the defaults: under 600 so it fires whichever way an endpoint reporting `timeout: 0` is read, and
  `3 * 570 = 1710 < 5400` so it does not silently raise `vivijure-core`'s phase stall ceiling
  (`max(PHASE_HARD_DEADLINE_SECONDS, FINISH_STEP_MAX_ATTEMPTS * longest declared)` FLOORS that
  ceiling; it does not cap it). `deploy.sh` passes the value to the container so the two cannot drift,
  and `Dockerfile.serve` pins it to `0` -- the homelab door has no platform kill, so it keeps the full
  1200 (`3 * 1200 = 3600`, still inside the floor).

- **fix(upscale): the guard's expiry RETURNS a soft degrade and cannot be swallowed (#105).** Expiry
  returns `{"ok": false, "detail": "<reason>"}` on every job path -- never a raise, which would leave
  no structured output and fail the film after the GPU spend is already banked, strictly worse than the
  hang this replaces. `detail` rather than `error` because `degradeReason` reads it first
  (`vivijure-cf` `modules/_shared/finish-soft-degrade.ts:77`) and a top-level `error` is lifted by
  RunPod into a FAILED envelope that books a `failed` job row for a job that degraded honestly.
  Ordinary errors keep `error` on purpose: they ARE failures. The reason carries the guard name, the
  step and the elapsed seconds inside the first 120 characters, which is all the panel keeps.

  `InvocationExpired` derives from **BaseException**, not Exception. Every job path in this file ends
  in a broad `except Exception`, and a guard raising an ordinary exception is caught there, re-keyed to
  the legacy shape, and reported as an ordinary failure: present, tested, and inert on the exact path
  it exists for. Nothing has to remember to re-raise it. `finally` blocks still run, so the #98
  abort-path CUDA cache release is unaffected and is still asserted.

  KNOWN LIMIT, not fixable from this repo: the `passthrough:` tag the panel builds is generic for every
  cause (`vivijure-core#226`), so a wall-clock degrade and a no-face degrade are indistinguishable
  downstream. The reason text carries the cause; the counted tag does not.

- **fix(deploy): export the optional knobs so their documented defaults actually apply (#105).**
  `deploy.sh` sets eight defaults (`CONTAINER_DISK_GB`, `WORKERS_MIN/MAX`, `IDLE_TIMEOUT`,
  `EXECUTION_TIMEOUT_MS`, ...) with bare shell assignments, then reads them through `os.environ` in a
  child `python3`. A bare assignment is not in a child's environment, so only the knobs an operator had
  set in `deploy.env` (exported by the `set -a` around the source) ever arrived: every default on those
  lines was inoperative and left the template/endpoint body raising `KeyError`. Found while wiring
  `EXECUTION_TIMEOUT_MS` through to the container.

## v1.1.2 -- 2026-08-14

- **fix(upscale): release the CUDA cache on the ABORT path, not only on success (#98).**
  `torch.cuda.empty_cache()` sat immediately AFTER the batch loop with no `try/finally`, so the
  `raise TimeoutError("upscale exceeded FFMPEG_TIMEOUT")` inside the loop jumped straight over it and
  the torch caching allocator kept its reservation for the LIFE OF THE PROCESS.

  MEASURED on fatmike, two throwaway containers from the same image, argv differing only in
  `FFMPEG_TIMEOUT`, each idled 120s with the container still running after the job reached a terminal
  state:

  | FFMPEG_TIMEOUT | terminal state | VRAM at terminal | VRAM after 120s idle | after container stop |
  |---|---|---|---|---|
  | 21600 | COMPLETED | 459 MiB | **459 MiB** | 202 MiB |
  | 30 | FAILED, `upscale exceeded FFMPEG_TIMEOUT` | 19771 MiB | **19771 MiB** | 202 MiB |

  **So the leak is a property of the ABORT path, not of torch and not of the model.** A completed
  x4plus job releases normally; a timed-out one holds 19771 MiB of a 20475 MiB card indefinitely,
  leaving roughly 700 MiB for the co-tenant speech door that shares the card on both GPU twins. The
  two halves of #98 are one defect: the timeout CAUSES the memory problem. A `finally` is the whole
  fix, and it covers every exit path rather than special-casing `TimeoutError`.

  The decode deadline exits before this block and deliberately does not release: at that point the
  only CUDA allocation is the model weights, which are allocated rather than cached, so
  `empty_cache()` would free nothing. That branch is asserted in the tests so the next reader does
  not have to guess whether it was considered.

- **perf(upscale): carry the settled tile into the next batch instead of re-searching from `TILE`
  (#98).** `_upscale_batch` began its OOM-shrink search at the module-level `TILE` on every batch,
  and the tile a batch settled on was captured only for reporting (`tile_min`). On `RealESRGAN_x4plus`
  at 720p the tile settles 512 -> 256, so every batch re-ran a forward pass already known to OOM,
  caught the CUDA error, emptied the cache and retried. MEASURED on fatmike: a 3s clip is 72 frames
  at `BATCH=16`, so five such cycles; a 30s shot would be 45. The settled tile is now threaded
  forward, clamped into `[TILE_FLOOR, TILE]` so a carried value can never widen the search past the
  configured ceiling. It only ever narrows, and a later batch may still shrink further.

  The clamp is a named function rather than an inline expression because the mutation pass showed
  that deleting it reddened no test at all: it was unreachable from any hermetic test while it lived
  inside `_upscale_batch`, which needs torch and numpy to run.

- **Also recorded from the same run, because no measurement in #98 had one:** the first `phase_s`
  breakdown of an x4plus job. `{"extract": 0.15, "upscale": 807.87, "encode": 0.96}` on a 3s 720p
  clip -- **99.9% of the wall clock is the upscale loop**. The unguarded encode phase is therefore
  not a factor in the ceiling, and decode is noise.

## v1.1.1 -- 2026-08-14

- **feat(serve): a log line per job accept and per terminal transition (cf#507).** For six days a
  healthy door and an unreachable door produced identical `docker logs` output: one startup banner.
  The serve overlay logged nothing per request, so "zero log lines on the door" was never evidence
  that the door had not been reached -- it was evidence that the container had nothing to say, and a
  whole investigation was spent reasoning about that absence. Two lines now: an ACCEPT line at
  `submit()` carrying the job id and the post-enqueue queue depth, and a TERMINAL line carrying id,
  status, whether the job actually ran, and elapsed. The accept line is deliberate rather than
  redundant: a terminal line alone cannot separate "the door never received this request" from "it
  received it and died before reaching a terminal state", and the accept line is what makes a
  MISSING terminal line mean something. Depth is the saturation signal the single FIFO worker
  otherwise gives no way to see. No `logging` import, no framework, no levels: the file goes from
  one `print(` to three. The payload is never logged in whole or in part, because it carries
  presigned R2 URLs and a bearer.
- **fix(serve): never emit a log line while holding the registry lock (cf#507).** `_retain_locked()`
  is lock-held by contract and called `_log` directly. `print(flush=True)` to a container's stdout
  is a BLOCKING write, so a stalled log consumer would have wedged the whole door -- `submit`, `get`
  and `cancel` all take that lock -- and it would have presented as a door that accepts nothing and
  answers nothing, with no line to say why. A change whose purpose is making silence meaningful must
  not add a new way to go silent. Lines are now staged under the lock and emitted by `_drain_logs()`
  with it released; the worker drains once per loop iteration, which covers all four of its terminal
  transitions by construction, and `cancel()` drains explicitly because it returns from inside the
  lock. The drain is idempotent so a missed call delays a line rather than losing it.

## v1.1.0 -- 2026-08-07

- **ci(serve): publish the `*-serve` overlay to GHCR on every release tag (#89 item 1, fc#1592).**
  Nothing built a serve image and nothing would have, so both live video doors ran HAND-BUILT local
  tags -- forbidden by our own rule that evidence must be about a SHA, and unrollable by anyone but
  the operator who built them. `build-image.yml` now builds and pushes `<version>-serve`,
  `<major>.<minor>-serve` and `sha-<short>-serve` from the SAME job as the release image, gated
  identically (a bare merge to main smoke-builds and does not publish). Same job because the
  overlay's `FROM` **is** the release image: the ordering is not optional, the base is already in
  the local daemon (one thin layer instead of a second tens-of-GB pull), and the published overlay
  and its base always come from one source tree. Ported from `vivijure-audio-upscale`, where the
  same gap was closed first.
  **Not a verbatim port, and the difference was measured before writing it:** that repo pins its
  CUDA base by TAG and therefore needs plain `docker build` plus host prep, while this one pins by
  DIGEST (`FROM runpod/pytorch@sha256:263d4144...`), which is exactly why `build-push-action` is
  correct here and converting it to match the sibling would have been the wrong reflex. The release
  step keeps buildx and now states `load: true` so the overlay's base being local is a CONTRACT
  rather than an accident of the default driver; the overlay itself uses plain `docker build`,
  because it must resolve its base from the LOCAL daemon and buildx would re-resolve the name
  against the registry -- which on the smoke path does not exist yet and on the tag path could
  return a different image than the one just built. The serve step carries the same two controls as
  the sibling (the base must be one of the tags THIS job built, whole-line matched, and it must
  already exist locally) and prints `derived N serve tags of M release tags` with a floor.
- **fix(serve): `UPSCALE_IMAGE` no longer carries a default (#89 item 2).** It pinned the literal
  `:1.0.3` while the current release was `:1.0.5` -- two releases of silent drift, whose failure
  mode is a door that WORKS on a stale base, the failure nobody investigates. CI now passes the tag
  it just built; a hand build must pass the arg. Proved with a control pair on real docker: no arg
  -> `rc=1`, `base name (${UPSCALE_IMAGE}) should not be blank`, refused at parse before any pull;
  `--build-arg UPSCALE_IMAGE=busybox:latest` -> `rc=0`, so the refusal is the missing arg and not a
  malformed Dockerfile. The resulting `InvalidDefaultArgInFrom` warning is the intended shape and is
  documented as such in `CLAUDE.md`.
- **fix(ci): the release tag name no longer interpolates into a shell script.** The allowlist sync
  step passed `${{ github.ref_name }}` **inside** a `run:` block, and a GitHub expression is
  substituted into the shell source before bash parses it, so a tag name carrying shell
  metacharacters would be executed with a privileged token in the environment. Now passed via `env:`
  and validated against `^v[0-9]+(\.[0-9]+)*$` before use, matching what
  `vivijure-audio-upscale` already does. Called out separately in the PR because it is a security
  fix that happens to live in the file the bake touches, not part of the bake.
- **docs: `CLAUDE.md` gains a homelab serve-overlay section**, including the measured VRAM behaviour
  that distinguishes this door from the speech one: this handler calls `torch.cuda.empty_cache()`
  after upscale, so it returns to **192 MiB between jobs** with a transient **6392 MiB** peak during
  one, and that peak is a property of the JOB rather than a resident footprint. Also records that a
  default `{"selftest": true}` runs an R2 leg unless an explicit `model` is passed without `r2`.


- **fix(serve): forward `selftest` to the wrapped handler instead of intercepting it (#88,
  fc#1592).** `runpod_http_serve.route` answered `POST /run {"selftest": true}` with a canned
  `{"ok": true, "selftest": true, "service": ...}` before the request ever reached
  `handler._selftest`. That handler is the documented deploy-verification GPU check -- it loads
  every shipped model, generates a real multi-second clip and upscales it on the card -- so the
  interception made **the deploy check structurally incapable of failing**: it returned `ok: true`
  on a box with no GPU, a broken model, a missing weight or a dead ffmpeg. Ported from
  `vivijure-audio-upscale`, where the identical intercept was found and fixed first (`45a9369`);
  after this change the two repos' `runpod_http_serve.py` are **code-identical**, proved by
  token-stream comparison rather than asserted (1826 tokens each, with a negative control showing
  a mutated stream compares unequal). `/health` is unchanged and remains the auth-free liveness
  probe -- deliberately not the thing that proves the card works.
- **test: the serve route had NO coverage at all, in either sibling.** Measured before writing
  any: `runpod_http_serve` appeared in `Dockerfile.serve` and `serve.py` and in **zero** test
  files. `tests/test_serve_route.py` now covers forwarding (both the top-level and `input`
  spellings of `selftest`, since the removed intercept checked both and half of it could return
  unnoticed), an ordinary job as the discrimination control, `/health` being auth-free AND
  reaching nothing, 401 on a wrong and on a missing token, 503 when no token is configured at all
  (a different finding from 401 -- collapsing them would hide a misconfigured door), and 404 for
  an unknown job id, which is what makes cross-door job-id affinity measurable. It exercises the
  SHIPPED module with no stubs, because `runpod_http_serve` is stdlib-only by design.
- **test: cover the explicit-model-without-`r2` selftest path, which nothing did.** The suite had
  explicit-model WITH `r2: true` and the sweep paths, but not the path a homelab door's deploy
  check actually takes. It asserts on a call RECORDER rather than on the absence of an `r2` key,
  because a missing key is also what you would see if the leg ran and returned nothing, and it
  carries a positive control proving the recorder fires when the flag IS passed.
- **Both changes were mutation-tested, not merely run.** Reintroducing the intercept turns both
  forwarding tests red by name and prints the named assertion message; making the R2 leg
  unconditional turns the new bucket-traffic test red by name; restoring both returns the suite to
  green. A guard nobody has watched go red is not yet a guard.
- **No image is published by this.** `build-image.yml` in this repo still builds only `Dockerfile`
  and there is still no `*-serve` bake (#89 item 1), so the doors currently running on the GPU
  boxes are hand-built tags and **keep the interception until a serve image is published and they
  are redeployed from it**. This lands the fix; it does not deploy it.

## v1.0.5 -- 2026-07-24

- **PATCH: the RunPod Hub listing corrected to production reality (#75).** `BLACKWELL_96` (the
  RTX PRO 6000 pool production actually runs on) added to `gpuIds`, `containerDiskInGb` raised
  20 -> 40 because 20 GB cannot hold the 10.7 GB-compressed image uncompressed, and the Hub test
  moved onto the card production proves daily.
- Listing metadata and docs only, no handler or runtime change. **The tag publishes `:1.0.5`, which
  is functionally identical to `:1.0.4`, and the production endpoint deliberately stays pinned to
  the proven `:1.0.4`. No repin.**
  (Backfilled 2026-07-28 from the v1.0.5 GitHub release; this file did not exist at the tag.)

## v1.0.4 -- 2026-07-23

- **fix(security): security close-out (#73)** -- hardened `Content-Length` parsing, plus the
  false-positive documentation for the findings that were not real.
  (Backfilled 2026-07-28 from the v1.0.4 GitHub release; this file did not exist at the tag.)

## v1.0.3 -- 2026-07-22

Security PATCH for the video-upscale RunPod worker. Bake `:1.0.3`; the endpoint template was
repinned after `build-image` went green.

- **fix(security): require `project` scope for shared-bucket R2 keys (#69).**
- **fix(security): split coverage so a fork's pytest run lacks OIDC (#67).**
- **fix(security): pin the fleet-chezmoi sync; host builds on `ubuntu-latest` (#68).**
- **ci:** adversarial security audit workflow.
  (Backfilled 2026-07-28 from the v1.0.3 GitHub release; this file did not exist at the tag.)

## v1.0.2 -- 2026-07-21

- **docs(hub): the RunPod Hub publish surface.** Docs-only patch cut so Hub, which indexes releases
  rather than commits, could index a release tree containing `.runpod/` beside the root
  `handler.py` / `Dockerfile` / `README.md`: `.runpod/hub.json` plus `tests.json`
  (`{"selftest": true}` on L4), `.runpod/README.md` recording the R2 env names (`R2_ENDPOINT_URL`),
  `THIRD_PARTY_MODELS.md`, and the Hub badge.
- No handler or image-recipe change. Closes the Hub gap on upscale#61.
  (Backfilled 2026-07-28 from the v1.0.2 GitHub release; this file did not exist at the tag.)

## v1.0.1 -- 2026-07-15

- **PATCH cut so production pins a semver image rather than `sha-6651f0c`.** The content had only
  ever been published under a SHA tag, and a SHA is not a defensible canonical prod pin.
- deps: the pip-minor-patch group (#53), which is the content that had been sitting behind
  `sha-6651f0c`; Dependabot grouping (#52); dual-host control-panel SEO (#51).
- GHCR: `ghcr.io/skyphusion-labs/vivijure-upscale:1.0.1` (also `:1.0`, `sha-<short>`).
  (Backfilled 2026-07-28 from the v1.0.1 GitHub release; this file did not exist at the tag.)

## v1.0.0 -- 2026-07-11

- **First stable release of the ESRGAN video-upscale finish module.** The GPU upscale satellite
  (exact 2x, tiled), output-verified end to end for Studio v1.0.0: a low-base seedance 480p
  intermediate `shot_finished.mp4` upscaled to `_up` at exact 2x.
- **No code change since v0.2.10**; cut onto the stable v1.0.0 line as part of the
  constellation-wide milestone, carrying only a README correction (#50). The tag builds and
  publishes the consumer image.
  (Backfilled 2026-07-28 from the v1.0.0 GitHub release; this file did not exist at the tag.)

---

## Pre-1.0 line

Rows below are reconstructed from the commit log: these tags were cut before the repo kept a
changelog and none of them carries GitHub release notes. They are recorded so the line is
continuous, at commit-subject fidelity rather than pretending to more detail than survives.
(All backfilled 2026-07-28.)

### v0.2.10 -- 2026-07-11

- Free the torch CUDA cache before the NVENC encode, for small-card headroom.
- Report the settled tile, and add selftest resolution and duration knobs (#30 evidence).
- x4plus `alloc-conf` plus a tile-shrink small-card fallback, and an R2 selftest leg.
- Digest-pin the base image (#27); aviation-grade dependabot, coverage and org check alignment
  (#31); dependency bumps across pip and the actions group.

### v0.2.9 -- 2026-07-06

- Bound the RRDB forward so x4plus cannot CUDA-OOM (#28).

### v0.2.8 -- 2026-07-06

- Stamp the param-hash sidecar **after** the artifact (#25).
- Notify skyphusion-search corpus-sync on push to `main` (#24).

### v0.2.7 -- 2026-07-02

- **Pin job-supplied R2 keys to the render key map before any bucket I/O (#21).** A caller-supplied
  key reaching storage unmapped is the class this closes.
- Unify the `build-image` workflow across the finish satellites (#22); move it to GitHub-hosted
  `ubuntu-latest` so forks are safe (#10).
- `SUPPORT.md` routing security reports to `security@skyphusion.org` (#20); minimal ruff plus
  `py_compile` gate (#15); NOTICE naming the module and the FFmpeg written source offer (#14);
  outsider-runnable one-script deploy docs (#16).

### v0.2.6 -- 2026-06-23

- Make the upscale loop GPU-bound: stream frames, batch, fp16 (#7).

### v0.2.5 -- 2026-06-23

- Sample valid `nvidia-smi` fields in the selftest GPU sampler.

### v0.2.4 -- 2026-06-23

- Rename the sampler stop flag: `self._stop` shadowed `Thread._stop`.

### v0.2.3 -- 2026-06-23

- First tag in this repo: the video2x RunPod upscale image scaffold, then recovery of the real CUDA
  Real-ESRGAN source from the `:0.2.2` image (#4), FFmpeg 7 runtime libs, GHA RunPod image build
  replacing Jenkins (#1), keeping the encode on the GPU with an NVENC resolution cap and a bounded
  wall clock, and the AGPL `NOTICE` (#5).

---

## Not a release: `v0.0.99`

`v0.0.99` (2026-07-20) is a **Plane C bring-up tag**, not a version in this line. It points at
`fix(ci): skip hosted disk reclaim on Plane C GPU runners (#58)`, a commit that sits between v1.0.1
and v1.0.2 on `main`, and it carries the CI work that routed tag image builds to the self-hosted GPU
pool. It has no GitHub release and deliberately gets none; the sibling satellites use
`v0.0.99-plane-c` for the same purpose. Do not read it as preceding v0.2.x.
