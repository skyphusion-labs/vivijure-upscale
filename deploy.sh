#!/usr/bin/env bash
# Deploy this Vivijure finish engine -- one script.
#
# What it does, in order: build the Docker image, push it to your registry, then create (or reuse)
# a RunPod Serverless endpoint that runs it. When it finishes it prints your endpoint id, which you
# paste into your Vivijure Studio to turn this finish engine on. See docs/deploy.md for the long
# version and what every setting means.
#
# HOW TO USE:
#   cp deploy.env.example deploy.env     # then open deploy.env and fill in your keys
#   ./deploy.sh                          # safe to re-run
#
# It is IDEMPOTENT (re-running reuses what already exists) and FAILS CLOSED (any error stops the
# whole run, so you never end up with half a deploy). It never contains a secret: every key comes
# from your own deploy.env, which is git-ignored.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

say()  { printf "\n==> %s\n" "$*"; }
info() { printf "    %s\n" "$*"; }
die()  { printf "\nERROR: %s\n" "$*" >&2; exit 1; }

# RunPod REST v2 (REST v1 retired 2026-11-15; fleet-chezmoi#2085). On v2 an endpoint carries its own
# image, env, disk and registry: a v2 template is applied once at create and keeps no link, so this
# script makes no template and updates the endpoint itself. Paths and fields are from RunPod's v2
# OpenAPI (https://api.runpod.io/v2/openapi.json). RUNPOD_API_BASE is for this script's own tests.
API="${RUNPOD_API_BASE:-https://api.runpod.io/v2}"

# ---- 0. load and check deploy.env -------------------------------------------
[ -f deploy.env ] || die "deploy.env not found. Run: cp deploy.env.example deploy.env  (then edit it)."
set -a; . ./deploy.env; set +a

need() { local v; eval "v=\${$1:-}"; [ -n "$v" ] || die "deploy.env: $1 is required but empty -- $2"; }
need RUNPOD_API_KEY "your RunPod API key (runpod.io -> Settings -> API Keys)"
need IMAGE          "the image ref to build and run, e.g. ghcr.io/<you>/<repo>:latest"
need ENDPOINT_NAME  "a name for the RunPod endpoint"
[ -n "${GPU_TYPE_IDS:-}${GPU_POOLS:-}" ] || die "deploy.env: set GPU_TYPE_IDS (which GPU card(s) to pin the endpoint to) or GPU_POOLS (RunPod v2 pools) -- see deploy.env.example"

# sensible defaults for the optional knobs
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-20}"
WORKERS_MIN="${WORKERS_MIN:-0}"          # 0 = scale to zero (pay nothing when idle)
WORKERS_MAX="${WORKERS_MAX:-2}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-5}"        # seconds a worker stays warm after a job before scaling down
EXECUTION_TIMEOUT_MS="${EXECUTION_TIMEOUT_MS:-600000}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_ENDPOINT="${SKIP_ENDPOINT:-0}"
CONTAINER_REGISTRY_AUTH_ID="${CONTAINER_REGISTRY_AUTH_ID:-}"
# EXPORTED, and that is a fix rather than tidying (#105). The blocks below read these through
# `os.environ` in a child python3, and a bare shell assignment is not in a child's environment. Only
# the knobs an operator SET in deploy.env were exported (by the `set -a` around the source above), so
# every default on the eight lines here was inoperative: leaving CONTAINER_DISK_GB or
# EXECUTION_TIMEOUT_MS unset made the template/endpoint body raise KeyError, not take the documented
# default. Found while wiring EXECUTION_TIMEOUT_MS through to the container below.
export CONTAINER_DISK_GB WORKERS_MIN WORKERS_MAX IDLE_TIMEOUT EXECUTION_TIMEOUT_MS
export SKIP_BUILD SKIP_ENDPOINT CONTAINER_REGISTRY_AUTH_ID

# a tiny JSON reader/writer so we do not depend on jq being installed (python3 ships on every box)
pyget() { python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get(sys.argv[1],"") if isinstance(d,dict) else "")' "$1"; }

say "Deploy $ENDPOINT_NAME  (image: $IMAGE)"

# ---- 1. build the image ------------------------------------------------------
if [ "$SKIP_BUILD" = "1" ]; then
  say "Step 1/4: build image -- SKIPPED (SKIP_BUILD=1)"
else
  say "Step 1/4: build the Docker image"
  command -v docker >/dev/null || die "docker not found -- install Docker, or set SKIP_BUILD=1 if the image is already pushed."
  docker build -t "$IMAGE" . || die "docker build failed"
  info "built $IMAGE"
fi

# ---- 2. push the image -------------------------------------------------------
if [ "$SKIP_BUILD" = "1" ]; then
  say "Step 2/4: push image -- SKIPPED (SKIP_BUILD=1)"
else
  say "Step 2/4: push the image to your registry"
  if [ -n "${REGISTRY_TOKEN:-}" ]; then
    REG_HOST="${IMAGE%%/*}"; [ "$REG_HOST" = "$IMAGE" ] && REG_HOST="docker.io"
    printf "%s" "$REGISTRY_TOKEN" | docker login "$REG_HOST" -u "${REGISTRY_USER:-oauth}" --password-stdin \
      || die "docker login to $REG_HOST failed"
    info "logged in to $REG_HOST"
  else
    info "no REGISTRY_TOKEN set -- assuming you already ran 'docker login' for this registry"
  fi
  docker push "$IMAGE" || die "docker push failed (are you logged in to the registry, and is the repo created?)"
  info "pushed $IMAGE"
fi

# ---- 3. RunPod template + endpoint ------------------------------------------
if [ "$SKIP_ENDPOINT" = "1" ]; then
  say "Step 3/4: RunPod endpoint -- SKIPPED (SKIP_ENDPOINT=1). Image is built and pushed."
  say "Done (image only)."
  exit 0
fi

command -v curl >/dev/null || die "curl not found -- install curl."
AUTH=(-H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json")

api() {  # method path [body]
  local method="$1" path="$2" body="${3:-}"
  if [ -n "$body" ]; then
    curl -fsS -X "$method" "${AUTH[@]}" -d "$body" "$API$path"
  else
    curl -fsS -X "$method" "${AUTH[@]}" "$API$path"
  fi
}

# 3. endpoint: one v2 endpoint named $ENDPOINT_NAME, created or updated in place with everything on it.
say "Step 3/4: RunPod endpoint"
EP_BODY="$(python3 -c '
import json,os,sys
# endpoint env: R2 finish-chain creds plus any handler tuning knobs you set in deploy.env.
env={}
for k in ("R2_ENDPOINT_URL","R2_BUCKET","R2_ACCESS_KEY_ID","R2_SECRET_ACCESS_KEY",
          "MAX_OUTPUT_LONG_EDGE","FFMPEG_TIMEOUT","UPSCALE_BATCH","UPSCALE_TILE","UPSCALE_TILE_FLOOR",
          "UPSCALE_FP16","PYTORCH_CUDA_ALLOC_CONF"):
    v=os.environ.get(k,"")
    if v: env[k]=v
# The container has to KNOW the platform ceiling to size its own wall-clock guard under it (#105), and
# the only way that number cannot drift is to derive it from the same value we set on the endpoint.
env["UPSCALE_PLATFORM_TIMEOUT"]=str(int(os.environ["EXECUTION_TIMEOUT_MS"])//1000)
# GPU type -> v2 pool, from the RunPod GPU catalog (list-gpu-types, read 2026-11-11). A pool holds
# several cards of one class. GPU_POOLS names pools directly and wins when set.
POOL={"NVIDIA H200":"HOPPER_141","NVIDIA B200":"BLACKWELL_180","NVIDIA B300 SXM6 AC":"BLACKWELL_288",
      "NVIDIA H100 80GB HBM3":"ADA_80_PRO","NVIDIA H100 NVL":"ADA_80_PRO","NVIDIA H100 PCIe":"ADA_80_PRO",
      "NVIDIA A100 80GB PCIe":"AMPERE_80","NVIDIA A100-SXM4-80GB":"AMPERE_80",
      "NVIDIA RTX PRO 6000 Blackwell Server Edition":"BLACKWELL_96",
      "NVIDIA RTX PRO 6000 Blackwell Workstation Edition":"BLACKWELL_96",
      "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition":"BLACKWELL_96",
      "NVIDIA L40":"ADA_48_PRO","NVIDIA L40S":"ADA_48_PRO","NVIDIA RTX 6000 Ada Generation":"ADA_48_PRO",
      "NVIDIA A40":"AMPERE_48","NVIDIA RTX A6000":"AMPERE_48",
      "NVIDIA GeForce RTX 5090":"ADA_32_PRO","NVIDIA GeForce RTX 4090":"ADA_24",
      "NVIDIA GeForce RTX 3090":"AMPERE_24","NVIDIA RTX A5000":"AMPERE_24","NVIDIA L4":"AMPERE_24"}
pools=[x.strip() for x in os.environ.get("GPU_POOLS","").split(",") if x.strip()]
if not pools:
    for g in [x.strip() for x in os.environ.get("GPU_TYPE_IDS","").split(",") if x.strip()]:
        if g not in POOL:
            sys.exit("no RunPod v2 pool is known for GPU type %r; set GPU_POOLS (e.g. BLACKWELL_96) in deploy.env" % g)
        if POOL[g] not in pools: pools.append(POOL[g])
b={"name":os.environ["ENDPOINT_NAME"],"image":os.environ["IMAGE"],"env":env,
   "disk":int(os.environ["CONTAINER_DISK_GB"]),"gpu":{"pools":pools,"count":1},
   "workers":{"min":int(os.environ["WORKERS_MIN"]),"max":int(os.environ["WORKERS_MAX"]),
              "idleTimeout":int(os.environ["IDLE_TIMEOUT"])},
   "scaling":{"type":"QUEUE_DELAY","queueDelay":4},
   "timeout":int(os.environ["EXECUTION_TIMEOUT_MS"])}
auth=os.environ.get("CONTAINER_REGISTRY_AUTH_ID","")
if auth: b["registry"]=auth
print(json.dumps(b))
' )" || die "could not build the endpoint settings (see the message above)"

# Find the endpoint by name across every v2 page. "More pages but no cursor" stops the run: guessing
# "not found" would create a duplicate endpoint.
info "looking for an existing endpoint named $ENDPOINT_NAME"
EP_ID=""; CURSOR=""; PAGES=0
while : ; do
  PAGE="$(api GET "/serverless?limit=1000${CURSOR:+&cursor=$CURSOR}")" || die "could not list RunPod endpoints"
  LINE="$(printf '%s' "$PAGE" | python3 -c '
import sys,json
name=sys.argv[1]
try: d=json.load(sys.stdin)
except Exception: d={}
items=d if isinstance(d,list) else (d.get("endpoints") or d.get("items") or d.get("data") or [])
hit=""
for e in items:
    if isinstance(e,dict) and e.get("name")==name:
        hit=e.get("id",""); break
pg=d.get("pagination") if isinstance(d,dict) else None
more="yes" if isinstance(pg,dict) and pg.get("hasNextPage") is True else "no"
nxt=(pg.get("nextCursor") or "") if isinstance(pg,dict) else ""
print(hit+"\t"+nxt+"\t"+more)
' "$ENDPOINT_NAME")"
  EP_ID="$(printf '%s' "$LINE" | cut -f1)"
  [ -n "$EP_ID" ] && break
  [ "$(printf '%s' "$LINE" | cut -f3)" = "yes" ] || break
  CURSOR="$(printf '%s' "$LINE" | cut -f2)"
  [ -n "$CURSOR" ] || die "RunPod's endpoint list said there were more pages but gave no cursor; not guessing"
  PAGES=$((PAGES + 1)); [ "$PAGES" -lt 1000 ] || die "RunPod's endpoint list did not end; not guessing"
done

if [ -n "$EP_ID" ]; then
  info "updating endpoint $ENDPOINT_NAME ($EP_ID): image, env, GPU pools, workers and timeout"
  api PATCH "/serverless/$EP_ID" "$EP_BODY" >/dev/null || die "endpoint update failed"
else
  info "creating endpoint $ENDPOINT_NAME"
  # v2 requires the request-routing type on create; it is fixed then, and an update does not take it.
  CREATE_BODY="$(printf "%s" "$EP_BODY" | python3 -c "import sys,json; o=json.load(sys.stdin); o[\"type\"]=\"QUEUE\"; print(json.dumps(o))")" || die "could not build the create body"
  EP_ID="$(api POST /serverless "$CREATE_BODY" | pyget id)"
  [ -n "$EP_ID" ] || die "could not read the new endpoint id"
  info "created endpoint $EP_ID"
fi

# ---- 3c. optional post-deploy verify: run the selftest sweep on the live endpoint ----
# OFF by default (set VERIFY=1 to opt in) because it spends GPU seconds. When on, it submits the
# model-less sweep selftest ({"selftest": true}) -- which exercises EVERY shipped model on the real
# card AND the R2 finish-contract round-trip (#26) -- and FAILS CLOSED if the sweep is not ok. The R2
# leg is REQUIRED only when this endpoint has R2 creds set (else its honest-skip is expected, not a
# failure). Tune the poll ceiling with VERIFY_TIMEOUT (default 900s).
VERIFY="${VERIFY:-0}"
if [ "$VERIFY" = "1" ]; then
  say "Step 3c/4: verify -- running the selftest sweep on $EP_ID (VERIFY=1)"
  RUN_API="https://api.runpod.ai/v2/$EP_ID"
  if [ -n "${R2_ENDPOINT_URL:-}" ] && [ -n "${R2_ACCESS_KEY_ID:-}" ]; then R2REQ=true; else R2REQ=false; fi
  info "R2 leg required: $R2REQ (true only when R2 creds are set on this endpoint)"
  JOB_ID="$(curl -fsS -X POST "${AUTH[@]}" -d "{\"input\":{\"selftest\":true,\"scale\":4,\"r2\":$R2REQ}}" "$RUN_API/run" | pyget id)"
  [ -n "$JOB_ID" ] || die "verify: could not submit the selftest job"
  VERIFY_TIMEOUT="${VERIFY_TIMEOUT:-900}"
  info "submitted selftest job $JOB_ID; polling up to ${VERIFY_TIMEOUT}s"
  deadline=$(( $(date +%s) + VERIFY_TIMEOUT ))
  RESP=""
  while : ; do
    RESP="$(curl -fsS "${AUTH[@]}" "$RUN_API/status/$JOB_ID" || true)"
    STATUS="$(printf '%s' "$RESP" | pyget status)"
    case "$STATUS" in
      COMPLETED) break ;;
      FAILED|CANCELLED|TIMED_OUT) die "verify: selftest job $STATUS -- $RESP" ;;
    esac
    [ "$(date +%s)" -lt "$deadline" ] || die "verify: selftest did not finish within ${VERIFY_TIMEOUT}s"
    sleep 5
  done
  OK="$(printf '%s' "$RESP" | python3 -c 'import sys,json;d=json.load(sys.stdin);print("1" if (d.get("output") or {}).get("ok") else "0")' 2>/dev/null || echo 0)"
  [ "$OK" = "1" ] || die "verify: selftest sweep did NOT pass -- full result: $RESP"
  info "verify PASSED: selftest sweep ok (every model + the R2 leg)"
fi

# ---- 4. done: how to wire it into the studio --------------------------------
say "Step 4/4: done. Your finish engine is live."
cat <<MSG

  RunPod endpoint id:  $EP_ID

  Next step -- turn it on in your Vivijure Studio:
  ${STUDIO_ENV_HINT:-Paste the endpoint id above into your Studio config for this finish module. See docs/deploy.md.}

  Scale-to-zero is on (workersMin=$WORKERS_MIN), so you pay nothing while it sits idle.
MSG
