# Security audit false positives

Documented dismissals for adversarial-audit (K2.7/K3) findings that are not actionable bugs in this repo's threat model.

## Presigned mode SSRF (video_url / hash_url)

**Finding:** Presigned homelab mode fetches job-supplied URLs without SSRF guard.

**Disposition:** False positive (operator homelab mode). Presigned mode is an explicit operator-controlled deployment path for homelab R2 bypass; prod finish uses R2 mode with bucket-scoped credentials and host suffix pins (same disposition as vivijure-musetalk #69/#73 and vivijure-audio-upscale).

**Evidence:** `handler.py` presigned branch; prod RunPod template uses R2 mode, not presigned URLs from untrusted submitters.

## Record

| Date | Audit | Finding | Rationale |
| --- | --- | --- | --- |
| 2026-07-23 | K3 verify ~18:04 | Presigned video_url SSRF | Operator homelab mode; prod uses R2 mode |
