"""Presigned URL SSRF gate + query redaction (no GPU / network)."""

import os
import socket
import sys
import types


def _stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


_stub("torch", cuda=types.SimpleNamespace(is_available=lambda: False), __version__="0-stub",
      inference_mode=lambda *a, **k: (lambda f: f))
_stub("numpy")
_stub("boto3", client=lambda *a, **k: None)
_stub("spandrel", ModelLoader=object)


class _HTTPAdapter:
    def __init__(self, *a, **k):
        pass

    def init_poolmanager(self, *a, **k):
        return None


class _Session:
    def __init__(self):
        self.last = None

    def mount(self, prefix, adapter):
        pass

    def request(self, method, url, **k):
        self.last = (method, url, k)
        return types.SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            iter_content=lambda n: [b"x"],
            __enter__=lambda self: self,
            __exit__=lambda *a: None,
        )


_adapters = types.ModuleType("requests.adapters")
_adapters.HTTPAdapter = _HTTPAdapter
sys.modules["requests.adapters"] = _adapters
_requests = _stub("requests", Session=_Session)
_requests.adapters = _adapters

_runpod = _stub("runpod")
_runpod.serverless = types.SimpleNamespace(start=lambda *a, **k: None)

os.environ.setdefault("R2_ENDPOINT_URL", "https://stub.r2")
os.environ.setdefault("R2_ACCESS_KEY_ID", "stub")
os.environ.setdefault("R2_SECRET_ACCESS_KEY", "stub")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import handler  # noqa: E402


def _public_addrinfo(host, port, *a, **k):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port))]


def test_url_error_rejects_http_and_private(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))])
    assert handler._url_error("http://evil.example/x", "video_url")
    assert handler._url_error("https://127.0.0.1/x", "video_url")
    assert "blocked" in handler._url_error("https://loop.example/x", "video_url")
    assert "blocked" in handler._url_error("https://localhost/x", "video_url")


def test_url_error_rejects_link_local_metadata(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443))])
    assert "blocked" in handler._url_error("https://metadata.example/latest", "video_url")


def test_url_error_accepts_public_https(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _public_addrinfo)
    assert handler._url_error("https://bucket.example/obj", "video_url") is None


def test_url_error_host_suffix_pin(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _public_addrinfo)
    monkeypatch.setattr(handler, "R2_URL_HOST_SUFFIX", ".r2.cloudflarestorage.com")
    assert handler._url_error("https://evil.example/x", "video_url")
    assert handler._url_error(
        "https://acct.r2.cloudflarestorage.com/obj", "video_url") is None


def test_pinned_https_connects_to_resolved_ip(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _public_addrinfo)
    sess = _Session()
    # raising=False: sibling modules may have stubbed requests without Session first.
    monkeypatch.setattr(handler.requests, "Session", lambda: sess, raising=False)
    handler._pinned_https("GET", "https://bucket.example/obj", timeout=1, stream=True)
    method, url, k = sess.last
    assert method == "GET"
    assert url.startswith("https://8.8.8.8/")
    assert k["headers"]["Host"] == "bucket.example"
    assert k["allow_redirects"] is False


def test_presigned_rejects_bad_url_before_io(monkeypatch):
    called = {"pin": 0}

    def boom(*a, **k):
        called["pin"] += 1
        raise AssertionError("_pinned_https must not run for rejected URLs")

    monkeypatch.setattr(handler, "_pinned_https", boom)
    out = handler.handler({
        "input": {
            "video_url": "http://169.254.169.254/latest",
            "output_url": "https://bucket.example/o",
        },
    })
    assert out["ok"] is False and "error" in out
    assert called["pin"] == 0


def test_redact_query_strips_presigned_tokens():
    leaked = (
        "403 Client Error: Forbidden for url: "
        "https://acct.r2.cloudflarestorage.com/obj?X-Amz-Signature=deadbeef&X-Amz-Credential=AKIA"
    )
    out = handler._redact_query(leaked)
    assert "deadbeef" not in out
    assert "AKIA" not in out
    assert "X-Amz-Signature" not in out
    assert "[redacted]" in out
    assert "https://acct.r2.cloudflarestorage.com/obj" in out


def test_redact_query_strips_labeled_path_query():
    leaked = "Max retries exceeded with url: /obj?X-Amz-Signature=deadbeef"
    out = handler._redact_query(leaked)
    assert "deadbeef" not in out
    assert "X-Amz-Signature" not in out
    assert "[redacted]" in out


def test_handler_error_redacts_presigned_query(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError(
            "403 for url: https://r2.example/clip.mp4?X-Amz-Signature=deadbeef"
        )

    monkeypatch.setattr(handler, "_url_error", lambda *a, **k: None)
    monkeypatch.setattr(handler, "_pinned_https", boom)
    out = handler.handler({
        "input": {
            "video_url": "https://bucket.example/v",
            "output_url": "https://bucket.example/o",
        },
    })
    assert out["ok"] is False
    assert "deadbeef" not in out["error"]
    assert "X-Amz-Signature" not in out["error"]
    assert "[redacted]" in out["error"]
