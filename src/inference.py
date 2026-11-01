"""The shared CPU-only timed pipeline; no ground-truth access."""
from src import runtime
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from src.models import build_model
from src.preprocessing import image_tensor
from src.utils import ROOT, sha256, write_json, read_json


class Predictor:
    def __init__(self, weights: str | Path):
        runtime.configure()
        self.path = Path(weights)
        if not self.path.is_file():
            raise FileNotFoundError(f'Missing trained weights: {weights}. Train and freeze a model first.')
        bundle = torch.load(self.path, map_location='cpu', weights_only=True)
        self.config = bundle['model']
        self.resolution = int(bundle['resolution'])
        self.threshold = float(bundle['threshold'])
        if not 32 <= self.resolution <= 256 or not 0 < self.threshold < 1:
            raise ValueError('Invalid inference settings')
        self.model = build_model(self.config)
        self.model.load_state_dict(bundle['state_dict'], strict=True)
        self.model.cpu().eval()
        self.size_bytes = os.path.getsize(self.path)

    @torch.inference_mode()
    def __call__(self, bgr256: np.ndarray) -> np.ndarray:
        x = image_tensor(bgr256).unsqueeze(0)
        if self.resolution != 256:
            x = F.interpolate(x, size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)
        logits = self.model(x)
        if logits.shape[-2:] != (256, 256):
            logits = F.interpolate(logits, size=(256, 256), mode='bilinear', align_corners=False)
        return (logits.sigmoid()[0, 0] >= self.threshold).to(torch.uint8).numpy()

    def timed(self, bgr256: np.ndarray) -> tuple[np.ndarray, float]:
        start = time.process_time()
        mask = self(bgr256)
        return mask, (time.process_time()-start)*1000


def verify_frozen(weights: str | Path) -> dict:
    meta_path = Path(weights).with_suffix('.freeze.json')
    meta = read_json(meta_path)
    if meta['sha256'] != sha256(weights):
        raise ValueError('Frozen weights changed since selection')
    for file, digest in meta.get('source_sha256', {}).items():
        if sha256(ROOT/file) != digest:
            raise ValueError(f'Inference source changed after freeze: {file}')
    return meta


def export_weights(path: str | Path, model: torch.nn.Module, config: dict,
                   resolution: int, threshold: float = .5) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Plain torch.save; no post-compression. Includes learned BN buffers.
    torch.save(dict(format_version=1, model=config, resolution=resolution,
                    threshold=threshold,
                    state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}), path)
