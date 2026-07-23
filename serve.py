#!/usr/bin/env python3
"""Homelab HTTP entry for video upscale on LOCAL_FINISH_UPSCALE_URL."""
import os

from handler import handler
from runpod_http_serve import run_serve

if __name__ == "__main__":
    run_serve(
        handler,
        service="vivijure-upscale-finish-upscale",
        port=int(os.environ.get("PORT", "8012") or "8012"),
    )
