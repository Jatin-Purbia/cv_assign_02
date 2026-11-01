"""Validate resolution changes of a trained run without reading either test set."""
from src import runtime
runtime.configure()
import argparse
import csv
from pathlib import Path
from src.inference import Predictor, export_weights
from src.evaluation import evaluate
from src.utils import read_json, write_json
from src.utils import sha256
import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--resolutions', type=int, nargs='+', default=[128, 160, 192, 256])
    p.add_argument('--resume', action='store_true', help='Reuse matching completed validation observations')
    args = p.parse_args()
    run = Path(args.run)
    metadata = read_json(run/'run.json')
    if metadata['config'].get('synthetic_smoke'):
        raise ValueError('Cannot select synthetic models')
    model = Predictor(run/'best.pth')
    cfg = metadata['config']
    root = Path(cfg['data_root'])/'DUTS-TR'
    table = Path('results/model_comparison.csv')
    with table.open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    for resolution in args.resolutions:
        if not 32 <= resolution <= 256:
            raise ValueError('Resolution must be 32..256')
        name = f'{run.name}_infer{resolution}'
        weights = run/f'inference_{resolution}.pth'
        val_path = run/f'inference_{resolution}_validation.json'
        if args.resume and weights.exists() and val_path.exists():
            existing = Predictor(weights)
            value = read_json(val_path)
            same = (existing.config == model.config and existing.resolution == resolution
                    and existing.threshold == model.threshold
                    and value['checkpoint_sha256'] == sha256(weights)
                    and value['manifest_sha256'] == sha256(cfg['val_manifest'])
                    and all(torch.equal(v, existing.model.state_dict()[k]) for k, v in model.model.state_dict().items()))
            if not same:
                raise ValueError('Cached resolution result does not match model/manifest')
        else:
            export_weights(weights, model.model, model.config, resolution, model.threshold)
            value = evaluate(weights, root/'DUTS-TR-Image', root/'DUTS-TR-Mask', val_path,
                             'validation', cfg['val_manifest'])
        rows = [r for r in rows if r['name'] != name]
        rows.append(dict(name=name, weights=str(weights), validation_path=str(val_path),
                         parameters=metadata['parameters'],
                         **{k: value[k] for k in ('miou', 'bf1', 'q', 'cpu_ms', 'size_bytes', 'ces')}))
    with table.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected = max(rows, key=lambda r: float(r['ces']))
    for key in ('miou', 'bf1', 'q', 'cpu_ms', 'ces'):
        selected[key] = float(selected[key])
    selected['size_bytes'] = int(selected['size_bytes'])
    write_json('results/selection.json', selected)


if __name__ == '__main__':
    main()
