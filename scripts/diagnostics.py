"""Untrained synthetic latency diagnostic. These are NOT DUTS quality or official timings."""
from src import runtime
runtime.configure()
import argparse
from pathlib import Path
import numpy as np
import torch
from src.inference import export_weights, Predictor
from src.models import build_model
from src.utils import environment, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='results/synthetic_latency.json')
    args = p.parse_args()
    torch.manual_seed(42)
    folder = Path('tmp/diagnostics')
    folder.mkdir(parents=True, exist_ok=True)
    image = np.random.default_rng(42).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    rows = []
    for kind in ('tiny', 'unet', 'depthwise'):
        cfg = dict(kind=kind, base=8, depth=3)
        model = build_model(cfg)
        file = folder/f'untrained_{kind}.pth'
        export_weights(file, model, cfg, 160)
        predictor = Predictor(file)
        predictor(image)
        times = [predictor.timed(image)[1] for _ in range(50)]
        rows.append(dict(kind=kind, parameters=sum(p.numel() for p in model.parameters()),
                         size_bytes=predictor.size_bytes, mean_cpu_ms=sum(times)/len(times), runs=len(times)))
    write_json(args.out, dict(label='UNTRAINED synthetic-image diagnostic, not official performance',
                             environment=environment(), resolution=160, observations=rows))
    print(rows)


if __name__ == '__main__':
    main()
