# AGENTS.md

## Cursor Cloud specific instructions

This is a thin RunPod GPU serverless wrapper (`handler.py`). The heavy ML deps in
`requirements.txt` (torch, spandrel, ...) import only on the card and are NOT
installable/runnable on this CPU VM; a full upscale run needs a RunPod GPU pod.

CPU-testable CI gate (`.github/workflows/ci.yml`) -- the unit tests stub the heavy
deps via `sys.modules`, so they run without a GPU:

- Set up a venv (`.venv`; `python3.12-venv` is installed by the environment update
  script, which also creates the venv and installs these tools):
  `python3 -m venv .venv && .venv/bin/pip install ruff==0.15.20 pytest==9.1.1`.
- `.venv/bin/ruff check --select E9,F .`
- `.venv/bin/python -m py_compile handler.py`
- `.venv/bin/python -m pytest -q tests/` (OOM-split / sidecar / tile-shrink routing)

Verified in this environment: ruff clean, py_compile OK, 23 tests passed.
