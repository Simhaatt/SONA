import numpy as np
from stream_processor import StreamProcessor, cola_window


def test_cola_window_sums_to_one():
    for block, hop in [(1024, 512), (1024, 256), (512, 128), (800, 400)]:
        w = cola_window(block, hop)
        acc = np.zeros(block + hop * 8)
        for k in range(8):
            acc[k * hop : k * hop + block] += w
        steady = acc[block:-block]
        assert np.allclose(steady, 1.0, atol=1e-6), (block, hop, steady.min(), steady.max())


def test_identity_reconstruction():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000).astype(np.float32)
    sp = StreamProcessor(block_size=1024, hop_size=512, process=lambda b: b)
    out = []
    # feed in irregular chunk sizes to mimic a soundcard
    i = 0
    for step in (300, 512, 128, 1000, 733):
        while i < len(x):
            out.extend(sp.push(x[i : i + step]))
            i += step
            break
    while i < len(x):
        out.extend(sp.push(x[i : i + 512]))
        i += 512
    out.append(sp.flush())
    y = np.concatenate([np.asarray(o) for o in out])
    n = min(len(x), len(y))
    # interior must reconstruct the input (COLA), ignore warm-up/tail ramps
    a, b = x[2048 : n - 2048], y[2048 : n - 2048]
    err = np.max(np.abs(a - b))
    assert err < 1e-4, f"reconstruction error too high: {err}"


def test_identity_reconstruction_75pct_overlap():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(12000).astype(np.float32)
    sp = StreamProcessor(block_size=1024, hop_size=256, process=lambda b: b)
    out = []
    for i in range(0, len(x), 256):
        out.extend(sp.push(x[i : i + 256]))
    out.append(sp.flush())
    y = np.concatenate([np.asarray(o) for o in out])
    n = min(len(x), len(y))
    a, b = x[2048 : n - 2048], y[2048 : n - 2048]
    assert np.max(np.abs(a - b)) < 1e-4


def test_gain_process_scales_output():
    x = np.ones(8000, dtype=np.float32)
    sp = StreamProcessor(block_size=512, hop_size=256, process=lambda b: 0.5 * b)
    out = []
    for i in range(0, len(x), 256):
        out.extend(sp.push(x[i : i + 256]))
    out.append(sp.flush())
    y = np.concatenate([np.asarray(o) for o in out])
    interior = y[1024 : len(x) - 1024]
    assert np.allclose(interior, 0.5, atol=1e-3)


def test_rtf_tracks_work():
    x = np.zeros(4000, dtype=np.float32)
    sp = StreamProcessor(block_size=512, hop_size=256, process=lambda b: b)
    for i in range(0, len(x), 256):
        list(sp.push(x[i : i + 256]))
    assert sp.n_blocks > 0
    assert sp.realtime_factor >= 0.0
