# Changelog -- vivijure-upscale

The ESRGAN video-upscale finish satellite. The image ships as a git-tag-driven release
(`v<X.Y.Z>`; CI publishes the GHCR consumer image on tag push). The tag is the version of record,
and a tag publishing an image is **not** the same act as production being pinned to it: repins are
manual and spend-gated, and several tags below deliberately ship an image nothing repins to. This
file records the why behind each release. Newest first.

## Unreleased

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
