"""Torch Dataset yielding fixed-length (noisy, clean) crops for supervised training."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.dataset import Sample


class TorchPairDataset(Dataset):
    """Wrap an iterable of numpy `Sample`s into fixed-length training crops.

    Crops are random during training (augmentation) and deterministic (leading crop) otherwise.
    Samples shorter than `crop_len` are zero-padded.
    """

    def __init__(self, samples: list[Sample], crop_len: int = 32000, train: bool = True,
                 seed: int = 0) -> None:
        self.samples = [s for s in samples if s.clean is not None]
        if not self.samples:
            raise ValueError("TorchPairDataset requires samples with clean references")
        self.crop_len = crop_len
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _crop(self, clean: np.ndarray, noisy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(noisy)
        if n <= self.crop_len:
            pad = self.crop_len - n
            return (np.pad(clean, (0, pad)), np.pad(noisy, (0, pad)))
        start = int(self.rng.integers(0, n - self.crop_len)) if self.train else 0
        sl = slice(start, start + self.crop_len)
        return clean[sl], noisy[sl]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        clean, noisy = self._crop(s.clean.astype(np.float32), s.noisy.astype(np.float32))
        return torch.from_numpy(noisy), torch.from_numpy(clean)
