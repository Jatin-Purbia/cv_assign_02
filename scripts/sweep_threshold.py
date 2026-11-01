"""Bounded decision-threshold selection on DUTS-TR validation only."""
from src import runtime
runtime.configure()
import argparse
import csv
from pathlib import Path
from src.inference import Predictor, export_weights
from src.evaluation import evaluate
from src.utils import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', default='results/selection.json')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[.4, .45, .5, .55, .6])
    args = parser.parse_args()
    selected = read_json(args.selection)
    source = Path(selected['weights'])
    model = Predictor(source)
    run = read_json(source.parent/'run.json')
    if run['config'].get('synthetic_smoke'):
        raise ValueError('Cannot select synthetic models')
    cfg = run['config']
    root = Path(cfg['data_root'])/'DUTS-TR'
    table = Path('results/model_comparison.csv')
    with table.open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    for threshold in args.thresholds:
        if not 0 < threshold < 1:
            raise ValueError('Threshold must lie strictly between zero and one')
        name = f"{selected['name']}_thr{threshold:g}"
        weights = source.parent/f'threshold_r{model.resolution}_{threshold:g}.pth'
        val_path = weights.with_suffix('.validation.json')
        export_weights(weights, model.model, model.config, model.resolution, threshold)
        value = evaluate(weights, root/'DUTS-TR-Image', root/'DUTS-TR-Mask', val_path,
                         'validation', cfg['val_manifest'])
        rows = [r for r in rows if r['name'] != name]
        rows.append(dict(name=name, weights=str(weights), validation_path=str(val_path),
                         parameters=run['parameters'],
                         **{k: value[k] for k in ('miou', 'bf1', 'q', 'cpu_ms', 'size_bytes', 'ces')}))
    with table.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda r: float(r['ces']))
    for key in ('miou', 'bf1', 'q', 'cpu_ms', 'ces'):
        best[key] = float(best[key])
    best['size_bytes'] = int(best['size_bytes'])
    write_json('results/selection.json', best)


if __name__ == '__main__':
    main()
