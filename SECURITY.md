# Security policy

## Supported versions

This is a rolling, single-`main`-branch project. Only the latest revision receives security fixes; if
you are on an older checkout or image tag, update to the newest one to pick them up.

## Reporting a vulnerability

Please do not open a public GitHub issue for a security problem. Report it privately to
**security@skyphusion.org**. If you would rather use GitHub, open the repository's **Security** tab and
click **"Report a vulnerability"** to file a private advisory that only you and the maintainers can
see.

Please include:

- A description of the issue and its impact
- Steps to reproduce, with a minimal example if possible
- The affected version (image tag or commit SHA if known)
- Any suggestions for a fix

What to expect:

- **Acknowledgment** within a reasonable window (target: 5 business days).
- A **fix** in the latest revision once we confirm the issue; time-sensitive reports should say so.
- **Credit** for your report when the fix ships, unless you would rather stay anonymous.

Please give us a chance to ship a fix before any public disclosure (target: up to 90 days for a
coordinated fix).

## Scope

This is the Vivijure video upscale module, a Real-ESRGAN-based finish container, invoked by the Vivijure studio and its render backend (the trusted control plane). It
is not a standalone service: work arrives from the control plane as a job, the container processes it,
and it writes artifacts back to object storage by key. Its one runtime credential is an R2
(S3-compatible) token scoped to the shared bucket, delivered through the environment.

In-scope vulnerabilities include:

- Path traversal or unsafe key handling that reads or writes outside the intended bucket prefix or job
  workspace.
- Server-side request forgery or arbitrary object access via attacker-influenced keys.
- Command or argument injection into a render step or shell-out (ffmpeg, model tooling) driven by job
  input.
- Leakage of the R2 credential.

Out of scope:

- Issues that require an already-compromised control plane or an already-leaked credential.
- Denial of service from intentionally expensive but well-formed jobs (render cost is the operator's
  concern; submit access is gated by the control plane).
- The security posture of the upstream model weights or third-party libraries themselves (report those
  to their projects), beyond how this container invokes them.

## Scope of reports

Security reports should concern this code and its runtime. Please do not send code, diffs, or excerpts
you do not have the rights to share.
