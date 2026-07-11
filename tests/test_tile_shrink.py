"""CPU coverage for the small-card tile-shrink fallback (#30): _shrink_on_oom halves the tile on a CUDA
out-of-memory and retries down to a floor, so a card too small for one frame at the default tile through a
heavy 4x model still finishes instead of hard-failing. The shrink is pure control flow, so it is exercised
here with a fake pass function + the captured OOM string -- genuine coverage, no GPU.
"""

import contextlib
import sys
import types


def _as_module(name, obj):
    m = types.ModuleType(name)
    for k in dir(obj):
        if not k.startswith("__"):
            setattr(m, k, getattr(obj, k))
    return m


def _load_handler():
    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = "0-stub"
    torch_mod.float16 = "f16"
    torch_mod.inference_mode = lambda *a, **k: (lambda f: f)
    torch_mod.autocast = lambda **k: contextlib.nullcontext()
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
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


# The literal message captured from the real x4plus OOM (same fixture as test_oom_split).
CAPTURED_OOM = ("CUDA out of memory. Tried to allocate 45.70 GiB. GPU 0 has a total capacity of "
                "94.97 GiB of which 22.85 GiB is free.")


def _pass_fn(fits_at_or_below):
    """A fake tile pass: OOMs at any tile LARGER than fits_at_or_below, succeeds at or below it."""
    calls = []

    def pass_fn(tile):
        calls.append(tile)
        if tile > fits_at_or_below:
            raise RuntimeError(CAPTURED_OOM)
        return f"out@{tile}"

    pass_fn.calls = calls
    return pass_fn


def test_fits_at_first_tile_no_shrink():
    h = _load_handler()
    cleaned = []
    pf = _pass_fn(fits_at_or_below=512)
    out = h._shrink_on_oom(pf, tile=512, floor=64, cleanup=lambda: cleaned.append(1))
    assert out == "out@512"
    assert pf.calls == [512]      # fit at once, no shrink
    assert cleaned == []          # cleanup only runs on an OOM


def test_shrinks_by_halving_until_it_fits():
    h = _load_handler()
    cleaned = []
    pf = _pass_fn(fits_at_or_below=128)  # OOMs at 512 and 256, fits at 128
    out = h._shrink_on_oom(pf, tile=512, floor=64, cleanup=lambda: cleaned.append(1))
    assert out == "out@128"
    assert pf.calls == [512, 256, 128]   # halved each OOM
    assert len(cleaned) == 2             # cache freed before each retry


def test_oom_at_floor_reraises():
    h = _load_handler()
    cleaned = []
    pf = _pass_fn(fits_at_or_below=0)   # never fits, even at the floor
    try:
        h._shrink_on_oom(pf, tile=512, floor=64, cleanup=lambda: cleaned.append(1))
        assert False, "expected the floor-tile OOM to re-raise"
    except RuntimeError as e:
        assert "out of memory" in str(e).lower()
    assert pf.calls == [512, 256, 128, 64]   # shrank to the floor then gave up
    assert len(cleaned) == 3                  # freed before each retry, not after the final raise


def test_non_oom_error_is_not_swallowed():
    h = _load_handler()

    def boom(tile):
        raise RuntimeError("some other failure")

    try:
        h._shrink_on_oom(boom, tile=512, floor=64)
        assert False, "a non-OOM RuntimeError must propagate, not shrink"
    except RuntimeError as e:
        assert "out of memory" not in str(e).lower()


def test_cleanup_optional():
    h = _load_handler()
    pf = _pass_fn(fits_at_or_below=256)   # one shrink, no cleanup passed
    out = h._shrink_on_oom(pf, tile=512, floor=64)
    assert out == "out@256"
    assert pf.calls == [512, 256]


def test_parse_res_valid_and_evened():
    h = _load_handler()
    assert h._parse_res("2560x1440") == (2560, 1440)
    assert h._parse_res("1281x721") == (1280, 720)   # forced even
    assert h._parse_res("640X480") == (640, 480)      # case-insensitive


def test_parse_res_bad_input_falls_back_to_720p():
    h = _load_handler()
    for bad in ("", "junk", "12x", "x720", "9x9", "99999x1", None):
        assert h._parse_res(bad) == (1280, 720)
