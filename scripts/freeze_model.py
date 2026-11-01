"""Export a validation-selected trained model and lock its provenance for final tests."""
from src import runtime
runtime.configure()
import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from src.inference import Predictor
from src.utils import ROOT, read_json, write_json, sha256


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', default='results/selection.json')
    p.add_argument('--out', default='weights/final_model.pth')
    args = p.parse_args()
    selected = read_json(args.selection)
    source = Path(selected['weights'])
    run = read_json(source.parent/'run.json')
    if run['config'].get('synthetic_smoke'):
        raise ValueError('Synthetic smoke weights cannot become the final submission model')
    val = read_json(selected.get('validation_path', str(source.parent/'validation.json')))
    if val['split'] != 'validation' or val['checkpoint_sha256'] != sha256(source):
        raise ValueError('Validation evidence does not match selected weights')
    if selected['ces'] != val['ces'] or val['cpu_ms'] > 50:
        raise ValueError('Invalid selection or CPU limit exceeded')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Refusing to replace frozen model')
    predictor = Predictor(source)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, out)
    write_json(out.with_suffix('.freeze.json'), dict(sha256=sha256(out),
               source_sha256={f: sha256(ROOT/f) for f in ('src/inference.py', 'src/preprocessing.py',
                                                         'src/models.py', 'src/runtime.py')},
               frozen_utc=datetime.now(timezone.utc).isoformat(), validation=val,
               selection=selected, run=run, statement='Selected using DUTS-TR validation only'))
    final_config = dict(run['config'])
    final_config.update(inference_resolution=predictor.resolution, threshold=predictor.threshold,
                        weights=str(out), frozen=True)
    write_json('configs/final.json', final_config)


if __name__ == '__main__':
    main()
