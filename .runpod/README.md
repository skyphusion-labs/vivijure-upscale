# RunPod Hub -- Vivijure Upscale

Hub listing config for the Vivijure video-upscale finish satellite.

## Required environment (finish-chain / R2 mode)

| Env key | What to put |
| --- | --- |
| `R2_ENDPOINT_URL` | `https://<account-id>.r2.cloudflarestorage.com` |
| `R2_ACCESS_KEY_ID` | Public half of an R2 API token |
| `R2_SECRET_ACCESS_KEY` | Secret half of that token |
| `R2_BUCKET` | Bucket shared with Vivijure Studio (default `vivijure`) |

**Name check:** this worker reads `R2_ENDPOINT_URL`. The main `vivijure-backend` listing uses
`R2_ENDPOINT` (no `_URL`).

## Hub test

`.runpod/tests.json` sends `{ "selftest": true }` (generates a short clip, upscales end to end).
No R2 credentials required. Prefer an **Ada** card with NVENC (**L4** / **L40S**).

## GPU and disk (source of truth: the production endpoint)

`hub.json` mirrors what the Vivijure production endpoint actually runs (`4q8idwbk6tyqbq`, running `ghcr.io/skyphusion-labs/vivijure-upscale:1.0.4`,
read from the RunPod API on 2026-07-25), so a Hub deployer gets the configuration we ourselves
prove every day:

- `gpuIds`: `ADA_24,ADA_48,BLACKWELL_96,BLACKWELL_180,-NVIDIA GeForce RTX 4090`. `BLACKWELL_96` (RTX PRO 6000, NVENC capable) is the pool production uses and was
  missing from the list entirely; the Ada pools stay first for cost, and `BLACKWELL_180` stays as a
  large fallback. The RTX 4090 exclusion is kept as it was.
- `containerDiskInGb`: `40` (raised from 20, which could not hold the image uncompressed). The image is 10.7 GB compressed.
- `tests.json` pins `NVIDIA RTX PRO 6000 Blackwell Server Edition`: the card production runs on, so a
  green Hub test means the same thing our own endpoint means.

Repin this section together with `hub.json` whenever the production endpoint moves pools or image.

Third-party model inventory: [THIRD_PARTY_MODELS.md](../THIRD_PARTY_MODELS.md).
