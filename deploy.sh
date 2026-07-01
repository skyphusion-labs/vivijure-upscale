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

API="https://rest.runpod.io/v1"

# ---- 0. load and check deploy.env -------------------------------------------
[ -f deploy.env ] || die "deploy.env not found. Run: cp deploy.env.example deploy.env  (then edit it)."
set -a; . ./deploy.env; set +a

need() { local v; eval "v=\${$1:-}"; [ -n "$v" ] || die "deploy.env: $1 is required but empty -- $2"; }
need RUNPOD_API_KEY "your RunPod API key (runpod.io -> Settings -> API Keys)"
need IMAGE          "the image ref to build and run, e.g. ghcr.io/<you>/<repo>:latest"
need ENDPOINT_NAME  "a name for the RunPod endpoint"
need GPU_TYPE_IDS   "which GPU(s) to pin the endpoint to (see deploy.env.example)"

# sensible defaults for the optional knobs
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-20}"
WORKERS_MIN="${WORKERS_MIN:-0}"          # 0 = scale to zero (pay nothing when idle)
WORKERS_MAX="${WORKERS_MAX:-2}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-5}"        # seconds a worker stays warm after a job before scaling down
EXECUTION_TIMEOUT_MS="${EXECUTION_TIMEOUT_MS:-600000}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_ENDPOINT="${SKIP_ENDPOINT:-0}"
CONTAINER_REGISTRY_AUTH_ID="${CONTAINER_REGISTRY_AUTH_ID:-}"

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

# 3a. template: reuse one with this name if it exists, else create it.
say "Step 3/4: RunPod template + endpoint"
TPL_NAME="${ENDPOINT_NAME}-template"
info "looking for an existing template named $TPL_NAME"
TPL_ID="$(api GET /templates | python3 -c '
import sys,json
name=sys.argv[1]
try: data=json.load(sys.stdin)
except Exception: data=[]
items=data.get("templates",data) if isinstance(data,dict) else data
for t in (items or []):
    if isinstance(t,dict) and t.get("name")==name:
        print(t.get("id","")); break
' "$TPL_NAME" 2>/dev/null || true)"

# build the template body (env carries R2 creds for the finish-chain R2 mode, if you set them)
TPL_BODY="$(python3 -c '
import json,os
# endpoint env: R2 finish-chain creds plus any handler tuning knobs you set in deploy.env.
env={}
for k in ("R2_ENDPOINT_URL","R2_BUCKET","R2_ACCESS_KEY_ID","R2_SECRET_ACCESS_KEY",
          "MAX_OUTPUT_LONG_EDGE","FFMPEG_TIMEOUT","UPSCALE_BATCH","UPSCALE_TILE","UPSCALE_FP16"):
    v=os.environ.get(k,"")
    if v: env[k]=v
b={"name":os.environ["TPL_NAME"],"imageName":os.environ["IMAGE"],"isServerless":True,
   "containerDiskInGb":int(os.environ["CONTAINER_DISK_GB"]),"category":"NVIDIA","env":env}
auth=os.environ.get("CONTAINER_REGISTRY_AUTH_ID","")
if auth: b["containerRegistryAuthId"]=auth
print(json.dumps(b))
' )"
export TPL_NAME

if [ -n "$TPL_ID" ]; then
  info "updating existing template $TPL_ID"
  api PATCH "/templates/$TPL_ID" "$TPL_BODY" >/dev/null || die "template update failed"
else
  info "creating template $TPL_NAME"
  TPL_ID="$(api POST /templates "$TPL_BODY" | pyget id)"
  [ -n "$TPL_ID" ] || die "could not read the new template id"
  info "created template $TPL_ID"
fi

# 3b. endpoint: reuse one with this name if it exists, else create it.
info "looking for an existing endpoint named $ENDPOINT_NAME"
EP_ID="$(api GET /endpoints | python3 -c '
import sys,json
name=sys.argv[1]
try: data=json.load(sys.stdin)
except Exception: data=[]
items=data.get("endpoints",data) if isinstance(data,dict) else data
for e in (items or []):
    if isinstance(e,dict) and e.get("name")==name:
        print(e.get("id","")); break
' "$ENDPOINT_NAME" 2>/dev/null || true)"

if [ -n "$EP_ID" ]; then
  info "endpoint $ENDPOINT_NAME already exists ($EP_ID) -- leaving its GPU/scaling as set"
  info "(RunPod does not let the API change an endpoint's GPU list after creation; use the console to re-pin)"
else
  # gpuTypeIds is a comma-separated list in deploy.env; turn it into a JSON array.
  EP_BODY="$(GPU_TYPE_IDS="$GPU_TYPE_IDS" TPL_ID="$TPL_ID" python3 -c '
import json,os
gpus=[g.strip() for g in os.environ["GPU_TYPE_IDS"].split(",") if g.strip()]
b={"name":os.environ["ENDPOINT_NAME"],"templateId":os.environ["TPL_ID"],"computeType":"GPU",
   "gpuTypeIds":gpus,"workersMin":int(os.environ["WORKERS_MIN"]),
   "workersMax":int(os.environ["WORKERS_MAX"]),"idleTimeout":int(os.environ["IDLE_TIMEOUT"]),
   "executionTimeoutMs":int(os.environ["EXECUTION_TIMEOUT_MS"]),"scalerType":"QUEUE_DELAY","scalerValue":4}
print(json.dumps(b))
' )"
  info "creating endpoint $ENDPOINT_NAME"
  EP_ID="$(api POST /endpoints "$EP_BODY" | pyget id)"
  [ -n "$EP_ID" ] || die "could not read the new endpoint id"
  info "created endpoint $EP_ID"
fi

# ---- 4. done: how to wire it into the studio --------------------------------
say "Step 4/4: done. Your finish engine is live."
cat <<MSG

  RunPod endpoint id:  $EP_ID

  Next step -- turn it on in your Vivijure Studio:
  ${STUDIO_ENV_HINT:-Paste the endpoint id above into your Studio config for this finish module. See docs/deploy.md.}

  Scale-to-zero is on (workersMin=$WORKERS_MIN), so you pay nothing while it sits idle.
MSG
