# Security audit false positives

Documented dismissals for adversarial-audit (K2.7/K3) findings that are not actionable bugs in this repo's threat model.

## Homelab presigned mode

When `R2_URL_HOST_SUFFIX` is unset, presigned GET/PUT still refuse non-https, localhost, and
private / loopback / link-local / metadata addresses, and connect to the validated IP with
redirects off. The suffix pin is the remaining operator knob: empty default is homelab
convenience (any public https host). Production RunPod templates set the suffix.

## Record

| Date | Audit | Finding | Rationale |
| --- | --- | --- | --- |
| 2026-07-23 | K3 verify ~18:04 | Presigned video_url SSRF | Closed: `_url_error` + `_pinned_https` (https, blocked addrs, optional suffix, no redirects) |
| 2026-07-23 | K3 verify ~18:04 | Unbounded in-memory decode memory DoS | GPU worker bounded by clip duration cap + operator job limits |
