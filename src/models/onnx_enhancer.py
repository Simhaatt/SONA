"""ONNX Runtime backend for exported SHRAVAN mask models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..utils.config import Config
from .torch_frontend import TorchFrontend


class OnnxNeuralEnhancer:
    """Use ONNX Runtime for mask estimation and the standard Torch DSP frontend."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        onnx_path: str | Path,
        threads: int = 2,
        name: str = "shravan_onnx",
    ) -> None:
        import onnxruntime as ort

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.cfg = Config(checkpoint["cfg"])
        stft = self.cfg.get_path("stft", {})
        bands = self.cfg.get_path("bands", {})
        self.tf = TorchFrontend(
            sr=int(self.cfg.get_path("sr", 16000)),
            n_fft=int(stft.get("n_fft", 512)),
            hop=stft.get("hop"),
            n_bands=int(bands.get("n_bands", 32)),
            band_scale=bands.get("scale", "bark"),
        )
        self.is_complex = self.cfg.get_path("model.kind", "") == "complex_gtcrn"
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(
            str(Path(onnx_path).resolve()),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.name = name
        self.device = "onnx-cpu"
        self.threads = int(threads)

    def enhance(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        if sr != self.tf.sr:
            raise ValueError(f"expected {self.tf.sr} Hz audio, received {sr} Hz")
        x = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).reshape(1, -1)
        with torch.no_grad():
            spectrum, features = self.tf.analyze(x)
        mask = self.session.run(["gain"], {"feats": features.numpy()})[0]
        mask_tensor = torch.from_numpy(mask)
        with torch.no_grad():
            if self.is_complex:
                output = self.tf.apply_band_complex(spectrum, mask_tensor, length=x.shape[-1])
            else:
                output = self.tf.apply_band_gain(spectrum, mask_tensor, length=x.shape[-1])
        return output.squeeze(0).numpy().astype(np.float32, copy=False)
