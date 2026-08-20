#!/usr/bin/env python3
"""Phase 1, part B step 2: score PESQ (+STOI/ESTOI) for every variant vs clean. RUN INSIDE WSL:

    MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 -u root -- /opt/pesqenv/bin/python \
        "/mnt/e/Rohit/FYP/noise cancellation/PERCEPTUAL_MP/scripts/phase1_pesq_wsl.py"

Reads pesq_wavs/{clean,fp32,uniform_int8,only_<layer>}/<uid>.wav, computes wideband PESQ + STOI +
ESTOI for each variant vs the clean reference.

Hardened after a real hang: the `pesq` package's C extension can spin forever on a pathological
signal (silence/NaN/extreme values -- plausible from an unstable quantized variant) without
raising an exception, so no try/except at the Python level can catch it. Each PESQ call now runs
under a SIGALRM timeout, sanity-checks the signal first, and progress is written to a
line-buffered log immediately (not just at exit) so a future hang is diagnosable in seconds, not
by inference from frozen /proc/<pid>/io counters. Resumable: skips (uid, method) pairs already in
the per-clip CSV.
"""
import csv
import os
import signal
import sys
import time

import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi

WAVS = os.environ.get(
    "PESQ_WAVS",
    "/mnt/e/Rohit/FYP/noise cancellation/PERCEPTUAL_MP/pesq_wavs",
)
SR = 16000
PESQ_TIMEOUT_S = 15
METHODS = [
    "fp32", "uniform_int8",
    "only_stem", "only_block0_d1", "only_block1_d2",
    "only_block2_d4", "only_block3_d8", "only_head",
]

LOG = open(os.path.join(WAVS, "phase1_pesq_progress.log"), "a", buffering=1)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", file=LOG)
    print(msg, flush=True)


class TimeoutError(Exception):
    pass


def _on_alarm(signum, frame):
    raise TimeoutError()


def load(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64)


def sane(x: np.ndarray) -> bool:
    """Reject signals that are known to make the pesq C extension hang or misbehave."""
    if x.size == 0 or not np.all(np.isfinite(x)):
        return False
    if float(np.sqrt(np.mean(x**2))) < 1e-6:  # effectively silent
        return False
    return True


def safe_pesq(c: np.ndarray, d: np.ndarray):
    if not (sane(c) and sane(d)):
        return None, "degenerate_signal"
    old_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(PESQ_TIMEOUT_S)
    try:
        return float(pesq(SR, c, d, "wb")), None
    except TimeoutError:
        return None, "timeout"
    except Exception as e:
        return None, str(e)[:100]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def main():
    clean_dir = os.path.join(WAVS, "clean")
    uids = sorted(os.path.splitext(f)[0] for f in os.listdir(clean_dir) if f.endswith(".wav"))
    log(f"scoring {len(uids)} clips x {len(METHODS)} methods, PESQ timeout={PESQ_TIMEOUT_S}s")

    pc_path = os.path.join(WAVS, "pesq_per_clip.csv")
    done = set()
    if os.path.exists(pc_path):
        with open(pc_path, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["uid"], r["method"]))
        log(f"resuming: {len(done)} (uid,method) pairs already scored")

    is_new = not os.path.exists(pc_path)
    out_f = open(pc_path, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=["uid", "method", "pesq", "stoi", "estoi", "pesq_error"])
    if is_new:
        writer.writeheader()
        out_f.flush()

    n_done, n_skip, n_fail = 0, 0, 0
    for i, uid in enumerate(uids):
        clean = load(os.path.join(clean_dir, uid + ".wav"))
        for m in METHODS:
            if (uid, m) in done:
                n_skip += 1
                continue
            p = os.path.join(WAVS, m, uid + ".wav")
            if not os.path.exists(p):
                continue
            deg = load(p)
            n = min(len(clean), len(deg))
            c, d = clean[:n], deg[:n]

            t0 = time.time()
            pe, err = safe_pesq(c, d)
            elapsed = time.time() - t0
            if elapsed > 3:
                log(f"  SLOW: {uid} {m} took {elapsed:.1f}s (err={err})")
            if err:
                n_fail += 1

            try:
                st = float(stoi(c, d, SR, extended=False))
                es = float(stoi(c, d, SR, extended=True))
            except Exception:
                st = es = None

            writer.writerow({"uid": uid, "method": m, "pesq": pe, "stoi": st, "estoi": es,
                              "pesq_error": err or ""})
            out_f.flush()
            n_done += 1

        if (i + 1) % 20 == 0:
            log(f"  {i+1}/{len(uids)} clips (scored {n_done}, skipped {n_skip}, pesq_fail {n_fail})")

    out_f.close()
    log(f"done: scored {n_done} new rows, skipped {n_skip} already-done, {n_fail} pesq failures/timeouts")

    # summary
    with open(pc_path, newline="") as f:
        per_clip = list(csv.DictReader(f))
    summ_path = os.path.join(WAVS, "pesq_summary.csv")
    log(f"\n{'method':16s} {'PESQ':>7s} {'STOI':>7s} {'ESTOI':>7s}  (n_ok)")
    with open(summ_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "pesq", "stoi", "estoi", "n_ok", "n_total"])
        for m in METHODS:
            vals = [r for r in per_clip if r["method"] == m]

            def mean(k, vals=vals):
                xs = [float(r[k]) for r in vals if r.get(k) not in (None, "", "None")]
                return sum(xs) / len(xs) if xs else float("nan")

            pe, st, es = mean("pesq"), mean("stoi"), mean("estoi")
            n_ok = sum(1 for r in vals if r.get("pesq") not in (None, "", "None"))
            w.writerow([m, pe, st, es, n_ok, len(vals)])
            log(f"{m:16s} {pe:7.4f} {st:7.4f} {es:7.4f}  ({n_ok}/{len(vals)})")

    log(f"\nwrote {pc_path}\nwrote {summ_path}")


if __name__ == "__main__":
    main()
