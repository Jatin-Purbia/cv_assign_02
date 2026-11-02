"""Train/evaluate only DUTS-TR candidates. Never reads DUTS-TE or captured data."""
from src import runtime
runtime.configure()
import argparse
import copy
import csv
from pathlib import Path
from src.evaluation import evaluate
from src.trainer import train
from src.utils import read_json, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/final_train.json')
    p.add_argument('--experiments', default='configs/reproduce_experiments.json')
    p.add_argument('--out', default='experiments/candidates')
    p.add_argument('--append', action='store_true', help='Preserve previous measured comparison rows')
    args = p.parse_args()
    baseline, trials = read_json(args.config), read_json(args.experiments)['trials']
    target = Path('results/model_comparison.csv')
    rows = []
    if args.append and target.exists():
        with target.open(encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
    for trial in trials:
        cfg = copy.deepcopy(baseline)
        cfg.update(trial['overrides'])
        folder = Path(args.out)/trial['name']
        if not (folder/'summary.json').exists():
            train(cfg, folder)
        elif read_json(folder/'run.json')['config'] != cfg:
            raise ValueError('Existing experiment has a different config')
        root = Path(cfg['data_root'])/'DUTS-TR'
        result = evaluate(folder/'best.pth', root/'DUTS-TR-Image', root/'DUTS-TR-Mask',
                          folder/'validation.json', 'validation', cfg['val_manifest'])
        rows = [r for r in rows if r['name'] != trial['name']]
        rows.append(dict(name=trial['name'], weights=str(folder/'best.pth'),
                         validation_path=str(folder/'validation.json'),
                         parameters=read_json(folder/'run.json')['parameters'],
                         **{k: result[k] for k in ('miou', 'bf1', 'q', 'cpu_ms', 'size_bytes', 'ces')}))
    target.parent.mkdir(exist_ok=True)
    with target.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected = dict(max(rows, key=lambda row: float(row['ces'])))
    for key in ('miou', 'bf1', 'q', 'cpu_ms', 'ces'):
        selected[key] = float(selected[key])
    selected['size_bytes'] = int(selected['size_bytes'])
    write_json('results/selection.json', selected)


if __name__ == '__main__':
    main()
