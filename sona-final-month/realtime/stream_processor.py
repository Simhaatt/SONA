"""Block-based overlap-add streaming for a waveform-in / waveform-out enhancer.

The SONA GTCRN enhancer is causal and cheap (RTF ~0.006-0.06 on CPU), so it can
run in a realtime loop by processing short overlapping blocks and overlap-adding
the results with a Hann window (constant-overlap-add, COLA). This module is pure
numpy and has no audio-hardware dependency, so it is unit-testable and reused by
both the live-mic demo and the offline wav simulation.

    sp = StreamProcessor(block_size=1024, hop_size=512, process=fn)
    for out_chunk in sp.push(mic_chunk):
        speaker.write(out_chunk)
    tail = sp.flush()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np


def cola_window(block_size: int, hop_size: int) -> np.ndarray:
    """Periodic Hann normalised so hop-shifted copies sum to 1 (exact COLA).

    Synthesis-side windowing only: push() windows each processed block once and
    overlap-adds. For an identity processor the output is the input times the
    steady-state sum of hop-shifted windows S[n]; dividing the window by S makes
    that sum unity, so identity reconstructs exactly for any hop <= block.
    """
    w = np.hanning(block_size + 1)[:-1].astype(np.float64)  # periodic Hann
    # Build the steady-state periodic sum, aligned to sample-0 (positions are
    # integer multiples of hop, matching how push() places blocks).
    total = 2 * (block_size // hop_size) + 3
    big = np.zeros(block_size + hop_size * total, dtype=np.float64)
    for k in range(total):
        big[k * hop_size : k * hop_size + block_size] += w
    off = hop_size * (total // 2)  # multiple of hop -> consistent alignment
    steady = big[off : off + block_size].copy()
    steady[steady < 1e-8] = 1.0
    return (w / steady).astype(np.float64)


@dataclass
class StreamProcessor:
    block_size: int = 1024
    hop_size: int = 512
    process: Callable[[np.ndarray], np.ndarray] | None = None
    _in_buf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    _ola: np.ndarray = field(init=False)
    _ready: int = field(default=0, init=False)  # samples of _ola fully summed
    _win: np.ndarray = field(init=False)
    n_blocks: int = field(default=0, init=False)
    proc_seconds: float = field(default=0.0, init=False)

    def __post_init__(self):
        if self.hop_size > self.block_size:
            raise ValueError("hop_size must be <= block_size")
        self._win = cola_window(self.block_size, self.hop_size)
        self._ola = np.zeros(self.block_size, dtype=np.float64)

    def _run(self, block: np.ndarray) -> np.ndarray:
        if self.process is None:
            return block
        import time
        t0 = time.perf_counter()
        out = np.asarray(self.process(block), dtype=np.float64)
        self.proc_seconds += time.perf_counter() - t0
        self.n_blocks += 1
        if out.shape[0] != self.block_size:
            # tolerate length drift from the model by trimming/padding
            fixed = np.zeros(self.block_size, dtype=np.float64)
            m = min(out.shape[0], self.block_size)
            fixed[:m] = out[:m]
            out = fixed
        return out

    def push(self, chunk: np.ndarray) -> Iterator[np.ndarray]:
        """Feed input samples; yield hop-sized output chunks as they complete."""
        self._in_buf = np.concatenate([self._in_buf, np.asarray(chunk, dtype=np.float64)])
        while self._in_buf.shape[0] >= self.block_size:
            block = self._in_buf[: self.block_size]
            processed = self._run(block) * self._win
            self._ola[: self.block_size] += processed
            out = self._ola[: self.hop_size].copy()
            # slide the OLA buffer left by hop, zero-fill the tail
            self._ola = np.concatenate([
                self._ola[self.hop_size :],
                np.zeros(self.hop_size, dtype=np.float64),
            ])
            self._in_buf = self._in_buf[self.hop_size :]
            yield out.astype(np.float32)

    def flush(self) -> np.ndarray:
        """Process any buffered tail (zero-padded) and return remaining output."""
        outs = []
        if self._in_buf.shape[0] > 0:
            pad = self.block_size - self._in_buf.shape[0] % self.block_size
            if pad != self.block_size:
                self._in_buf = np.concatenate([self._in_buf, np.zeros(pad)])
            for out in self.push(np.zeros(0)):
                outs.append(out)
        outs.append(self._ola[: self.hop_size].astype(np.float32))
        return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)

    @property
    def realtime_factor(self) -> float:
        processed = self.n_blocks * self.hop_size
        if processed == 0:
            return 0.0
        # assume 16 kHz unless caller rescales; RTF is proc_time / audio_time
        return self.proc_seconds / (processed / 16000.0)
