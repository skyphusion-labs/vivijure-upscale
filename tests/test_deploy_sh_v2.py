"""deploy.sh on RunPod REST v2 (fleet-chezmoi#2085).

The real deploy.sh runs with SKIP_BUILD=1 against a stub `curl` on PATH, so no image is built and no
RunPod call is made. The stub logs "<METHOD> <url>" and the -d body, and answers by STUB_MODE. On v2 an
endpoint carries its own image, env, disk and registry: deploy.sh makes no template, finds the endpoint
by name across every page, then PATCHes it in place or POSTs a new one at /v2/serverless."""
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

FAKE_CURL = r'''#!/usr/bin/env bash
method=GET; url=""; body=""
while [ $# -gt 0 ]; do
  case "$1" in
    -X) method="$2"; shift 2 ;;
    -d) body="$2"; shift 2 ;;
    -H) shift 2 ;;
    -*) shift ;;
    *) url="$1"; shift ;;
  esac
done
echo "$method $url" >> "$STUB_LOG"
[ -n "$body" ] && printf "%s\n" "$body" >> "$STUB_BODIES"
last='"pagination":{"nextCursor":null,"hasNextPage":false}'
case "$STUB_MODE:$method $url" in
  "fresh:GET "*/serverless*) printf '{"endpoints":[],%s}' "$last" ;;
  "page2:GET "*cursor=p2*) printf '{"endpoints":[{"id":"ep-2","name":"my-endpoint"}],%s}' "$last" ;;
  "page2:GET "*/serverless*) printf '{"endpoints":[{"id":"ep-1","name":"other"}],"pagination":{"nextCursor":"p2","hasNextPage":true}}' ;;
  "nocursor:GET "*/serverless*) printf '{"endpoints":[],"pagination":{"nextCursor":null,"hasNextPage":true}}' ;;
  *":POST "*/serverless) printf '{"id":"ep-created"}' ;;
  *":PATCH "*/serverless/*) printf '{"id":"ep-2"}' ;;
  *) echo "stub: unexpected $method $url" >&2; exit 22 ;;
esac
'''

ENV = """RUNPOD_API_KEY=stub
IMAGE=ghcr.io/example/repo:1.2.3
ENDPOINT_NAME=my-endpoint
GPU_TYPE_IDS="NVIDIA RTX PRO 6000 Blackwell Server Edition"
SKIP_BUILD=1
R2_BUCKET=b
CONTAINER_REGISTRY_AUTH_ID=reg-1
"""


def _run(tmp_path, mode, extra=""):
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(ROOT / "deploy.sh", work / "deploy.sh")
    (work / "deploy.env").write_text(ENV + extra)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    curl = bindir / "curl"
    curl.write_text(FAKE_CURL)
    curl.chmod(curl.stat().st_mode | stat.S_IEXEC)
    log, bodies = tmp_path / "calls.log", tmp_path / "bodies.log"
    log.write_text("")
    bodies.write_text("")
    env = {"PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}", "STUB_LOG": str(log),
           "STUB_BODIES": str(bodies), "STUB_MODE": mode, "HOME": str(tmp_path)}
    r = subprocess.run(["bash", str(work / "deploy.sh")], cwd=work, env=env,
                       capture_output=True, text=True, timeout=60)
    sent = [json.loads(b) for b in bodies.read_text().splitlines() if b.strip()]
    return r, log.read_text().splitlines(), sent


def test_fresh_deploy_posts_one_v2_endpoint_carrying_everything(tmp_path):
    r, calls, sent = _run(tmp_path, "fresh")
    assert r.returncode == 0, r.stdout + r.stderr
    assert calls == ["GET https://api.runpod.io/v2/serverless?limit=1000",
                     "POST https://api.runpod.io/v2/serverless"]
    assert not any("/templates" in c for c in calls)
    assert {urlsplit(c.split(" ", 1)[1]).hostname for c in calls} == {"api.runpod.io"}
    b = sent[0]
    assert b["type"] == "QUEUE"
    assert b["name"] == "my-endpoint" and b["image"] == "ghcr.io/example/repo:1.2.3"
    assert b["gpu"] == {"pools": ["BLACKWELL_96"], "count": 1}
    assert b["workers"]["min"] == 0 and "max" in b["workers"] and "idleTimeout" in b["workers"]
    assert b["scaling"] == {"type": "QUEUE_DELAY", "queueDelay": 4}
    assert isinstance(b["timeout"], int) and b["timeout"] > 0
    assert b["registry"] == "reg-1"
    assert b["env"].get("R2_BUCKET") == "b"


def test_an_existing_endpoint_on_a_later_page_is_updated_in_place(tmp_path):
    r, calls, sent = _run(tmp_path, "page2")
    assert r.returncode == 0, r.stdout + r.stderr
    assert calls == ["GET https://api.runpod.io/v2/serverless?limit=1000",
                     "GET https://api.runpod.io/v2/serverless?limit=1000&cursor=p2",
                     "PATCH https://api.runpod.io/v2/serverless/ep-2"]
    assert sent[0]["image"] == "ghcr.io/example/repo:1.2.3"
    assert "type" not in sent[0], "type is create-only on v2"


def test_a_list_that_claims_more_without_a_cursor_stops_before_any_write(tmp_path):
    r, calls, _ = _run(tmp_path, "nocursor")
    assert r.returncode != 0
    assert "gave no cursor" in r.stdout + r.stderr
    assert not any(c.startswith(("POST ", "PATCH ")) for c in calls)


def test_an_unmapped_gpu_type_stops_before_any_call(tmp_path):
    r, calls, _ = _run(tmp_path, "fresh", 'GPU_TYPE_IDS="NVIDIA Z9"\n')
    assert r.returncode != 0
    assert "GPU_POOLS" in r.stdout + r.stderr
    assert calls == []


def test_gpu_pools_can_be_named_directly(tmp_path):
    r, _, sent = _run(tmp_path, "fresh", "GPU_POOLS=ADA_48_PRO,BLACKWELL_96\n")
    assert r.returncode == 0, r.stdout + r.stderr
    assert sent[0]["gpu"]["pools"] == ["ADA_48_PRO", "BLACKWELL_96"]


def test_no_shipped_file_names_the_v1_host():
    host = "rest." + "runpod.io"
    files = [ROOT / "deploy.sh"] + [p for p in ROOT.rglob("*.py") if "tests" not in p.parts and ".git" not in p.parts]
    assert any(f.name == "deploy.sh" for f in files)
    assert [f.relative_to(ROOT).as_posix() for f in files if host in f.read_text(errors="replace")] == []
