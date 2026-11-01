"""Render two lowest-quality validation observations for manual evidence-based review."""
from src import runtime
runtime.configure()
import argparse
from pathlib import Path
from src.utils import read_json, write_json, sha256
from src.visualization import overlays


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', default='results/validation_metrics.json')
    p.add_argument('--images', default='data/DUTS/DUTS-TR/DUTS-TR-Image')
    p.add_argument('--masks', default='data/DUTS/DUTS-TR/DUTS-TR-Mask')
    p.add_argument('--out', default='results/failure_figures')
    args = p.parse_args()
    result = read_json(args.result)
    if result['split'] != 'validation':
        raise ValueError('Use validation observations for failure selection, not test-set tuning')
    if sha256(result['checkpoint']) != result['checkpoint_sha256']:
        raise ValueError('Evaluated checkpoint changed')
    chosen = sorted(result['per_image'], key=lambda r: r['q'])[:2]
    if len(chosen) < 2:
        raise ValueError('Need at least two evaluated images')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/'ids.txt'
    manifest.write_text('\n'.join(r['id'] for r in chosen)+'\n', encoding='utf-8')
    records = overlays(result['checkpoint'], args.images, args.masks, out, manifest)
    write_json(out/'review.json', dict(selection='lowest per-image validation Q',
               records=records, status='Inspect photographs and explain actual causes before report inclusion'))


if __name__ == '__main__':
    main()
