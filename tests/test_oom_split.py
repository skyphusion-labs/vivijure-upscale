"""CPU coverage for _forward_tile batch-split on CUDA OOM (no GPU, torch stubbed).

The real defect (#584 sib): RealESRGAN_x4plus (heavy RRDB, native 4x) on a 16-frame batch of a
near-full-frame tile allocated ~46 GiB in one forward and failed every real upscale job. _forward_tile
must SPLIT the batch on a CUDA out-of-memory RuntimeError -- recursing on halves down to a single frame,
emptying the cache between tries -- so a heavy model can never hard-OOM. The forward itself is GPU work,
but the split/recurse control flow under test is pure and is exercised here with a fake model + fake
tensors, so this is genuine coverage, not fabricated GPU coverage.
"""

import contextlib
import sys
import types


class FakeT:
    """Minimal stand-in for a batched tensor: carries the batch size and slices on the batch dim."""
    def __init__(self, n):
        self.n = n
        self.shape = (n, 3, 8, 8)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop if key.stop is not None else self.n
            return FakeT(stop - start)
        raise KeyError(key)

    def float(self):
        return self


def _stub_torch(max_batch):
    """Install a torch stub whose model OOMs above max_batch; returns (torch_module, model)."""
    calls = []

    def model(t):
        calls.append(t.n)
        if t.n > max_batch:
            raise RuntimeError("CUDA out of memory. Tried to allocate 46.00 GiB")
        return FakeT(t.n)

    model.calls = calls
    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = "0-stub"
    torch_mod.float16 = "f16"
    torch_mod.inference_mode = lambda *a, **k: (lambda f: f)
    torch_mod.autocast = lambda **k: contextlib.nullcontext()
    torch_mod.cat = lambda parts, dim=0: FakeT(sum(p.n for p in parts))
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    return torch_mod, model


def _load_handler(torch_mod):
    for name, mod in {
        "torch": torch_mod,
        "boto3": types.SimpleNamespace(client=lambda *a, **k: None),
        "numpy": types.ModuleType("numpy"),
        "requests": types.ModuleType("requests"),
        "spandrel": types.SimpleNamespace(ModelLoader=object),
    }.items():
        sys.modules[name] = mod if isinstance(mod, types.ModuleType) else _as_module(name, mod)
    runpod = types.ModuleType("runpod")
    runpod.serverless = types.SimpleNamespace(start=lambda *a, **k: None)
    sys.modules["runpod"] = runpod
    sys.modules.pop("handler", None)
    import handler
    return handler


def _as_module(name, obj):
    m = types.ModuleType(name)
    for k in dir(obj):
        if not k.startswith("__"):
            setattr(m, k, getattr(obj, k))
    return m


def test_fits_in_one_forward_no_split():
    torch_mod, model = _stub_torch(max_batch=16)
    h = _load_handler(torch_mod)
    out = h._forward_tile(model, FakeT(16), use_half=False)
    assert out.n == 16
    assert model.calls == [16]  # fit at once, no split


def test_splits_batch_until_it_fits():
    torch_mod, model = _stub_torch(max_batch=4)  # OOMs above 4 frames, like x4plus at full batch
    h = _load_handler(torch_mod)
    out = h._forward_tile(model, FakeT(16), use_half=False)
    assert out.n == 16  # every frame still processed
    ok_calls = [c for c in model.calls if c <= 4]
    assert sum(ok_calls) == 16  # the successful sub-forwards cover all 16 frames
    assert all(c <= 4 for c in ok_calls)
    assert len(model.calls) > 1  # it actually split


def test_single_frame_that_cannot_fit_reraises():
    torch_mod, model = _stub_torch(max_batch=0)  # OOMs even at 1 frame
    h = _load_handler(torch_mod)
    try:
        h._forward_tile(model, FakeT(1), use_half=False)
        assert False, "expected a re-raised OOM"
    except RuntimeError as e:
        assert "out of memory" in str(e).lower()


def test_non_oom_runtime_error_is_not_swallowed():
    torch_mod, _ = _stub_torch(max_batch=16)

    def boom(t):
        raise RuntimeError("some other failure")

    h = _load_handler(torch_mod)
    try:
        h._forward_tile(boom, FakeT(8), use_half=False)
        assert False, "a non-OOM RuntimeError must propagate, not split"
    except RuntimeError as e:
        assert "out of memory" not in str(e).lower()
