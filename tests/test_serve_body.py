"""#106: live-server coverage that _body() produces the BodyRejected sentinel.

route() unit tests in test_serve_route.py prove the sentinel is refused AFTER
auth. They cannot see _body(): it is a closure inside run_serve(). This file
drives a real HTTP server in a SUBPROCESS (run_serve installs a SIGTERM handler,
which raises ValueError anywhere but the main thread) so a regression that
returns None again fails here even if the route() tests stay green.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)

import runpod_http_serve as S  # noqa: E402

TOKEN = "test-token-not-a-real-secret"

SERVER_SRC = '''
import json, sys
sys.path.insert(0, {repo!r})
import runpod_http_serve as S

record = {record!r}

def handler(job):
    with open(record, "a") as fh:
        fh.write(json.dumps(job.get("input")) + "\\n")
    return {{"ok": True}}

S.run_serve(handler, service="test", host="127.0.0.1", port={port})
'''


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("serve")
    record = str(tmp / "received.jsonl")
    script = tmp / "server.py"
    port = _free_port()
    script.write_text(SERVER_SRC.format(repo=REPO, record=record, port=port))
    env = dict(os.environ, LOCAL_FINISH_TOKEN=TOKEN)
    proc = subprocess.Popen([sys.executable, str(script)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    deadline = time.time() + 20
    while time.time() < deadline:
        if proc.poll() is not None:
            pytest.fail("server exited early: %s" % (proc.stdout.read() or b"").decode()[:2000])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
            break
        except Exception:
            time.sleep(0.05)
    else:
        proc.kill()
        pytest.fail("server never came up")

    yield port, record
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _received(record):
    if not os.path.isfile(record):
        return []
    with open(record) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _await_one_more(record, before, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        got = _received(record)
        if len(got) > before:
            return got
        time.sleep(0.02)
    return _received(record)


def _post(port, raw: bytes, token=TOKEN):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/run", data=raw, method="POST",
        headers={"content-type": "application/json", "authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _post_claimed_length(port, raw: bytes, claimed_length: int, token=TOKEN):
    """POST with a Content-Length that is not the bytes we actually send.

    _body() decides oversize from the header and returns BEFORE rfile.read, so
    a live over-cap test must not ship a 1 MiB+ body: the door closes without
    reading and the client is reset mid-send. Lying about the length is the
    actual decision the cap makes.
    """
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    try:
        conn.putrequest("POST", "/run")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Authorization", f"Bearer {token}")
        conn.putheader("Content-Length", str(claimed_length))
        conn.endheaders()
        if raw:
            conn.send(raw)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read() or b"{}")
    finally:
        conn.close()


def test_body_under_the_cap_is_delivered(live_server):
    """POSITIVE CONTROL: without this, the over-cap test could pass against a dead door."""
    port, record = live_server
    before = len(_received(record))
    status, payload = _post(port, json.dumps({"input": {"project": "small"}}).encode())
    assert status == 200 and "id" in payload
    got = _await_one_more(record, before)
    assert len(got) > before, "under-cap job never reached the handler"
    assert got[-1] == {"project": "small"}, f"under-cap body not delivered: {got[-1]}"


def test_body_over_the_cap_is_413_not_an_empty_job(live_server):
    port, record = live_server
    before = len(_received(record))
    claimed = S.MAX_HTTP_BODY_BYTES + 1
    status, payload = _post_claimed_length(port, b'{"input":{"project":"x"}}', claimed)
    assert status == 413, f"over-cap accepted as a job: {status} {payload}"
    assert payload.get("ok") is False
    assert "exceeds" in payload.get("error", "")
    time.sleep(0.1)
    got = _received(record)
    assert len(got) == before, f"over-cap body was submitted: {got[before:]}"


def test_unparseable_body_is_400_not_an_empty_job(live_server):
    port, record = live_server
    before = len(_received(record))
    status, payload = _post(port, b"this is not json {")
    assert status == 400, f"unparseable body accepted as a job: {status} {payload}"
    assert payload.get("ok") is False
    assert "not valid JSON" in payload.get("error", "")
    time.sleep(0.1)
    got = _received(record)
    assert len(got) == before, f"unparseable body was submitted: {got[before:]}"


def test_oversize_without_a_token_is_401_over_a_real_socket(live_server):
    port, _ = live_server
    claimed = S.MAX_HTTP_BODY_BYTES + 1
    status, payload = _post_claimed_length(
        port, b'{"input":{"project":"x"}}', claimed, token="wrong-token",
    )
    assert status == 401, f"unauthenticated oversize disclosed the cap: {status} {payload}"
    assert payload.get("error") == "unauthorized"
