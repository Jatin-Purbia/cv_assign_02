"""Explicitly recorded PyTorch memory layout; the timed base pipeline is inherited."""
from pathlib import Path
from src.inference import Predictor
import torch


class ConfiguredPredictor(Predictor):
    def __init__(self, weights: str | Path):
        super().__init__(weights)
        metadata = torch.load(self.path, map_location='cpu', weights_only=True)
        self.memory_format = metadata.get('memory_format', 'contiguous')
        if self.memory_format not in ('contiguous', 'channels_last'):
            raise ValueError('Unsupported checkpoint memory format')
        if self.memory_format == 'channels_last':
            self.model.to(memory_format=torch.channels_last)


def export_configured_weights(path: Path, model: torch.nn.Module, config: dict,
                              resolution: int, threshold: float, memory_format: str) -> None:
    if memory_format not in ('contiguous', 'channels_last'):
        raise ValueError('Unsupported checkpoint memory format')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(dict(format_version=1, model=config, resolution=resolution, threshold=threshold,
                    memory_format=memory_format,
                    state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}), temporary)
    temporary.replace(path)
