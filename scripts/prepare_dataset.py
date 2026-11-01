"""Validate official DUTS-TR and create deterministic manifests without touching TE."""
from src import runtime
runtime.configure()
import argparse
from pathlib import Path
from src.dataset import paired_files, split_ids
from src.preprocessing import read_pair
from src.utils import sha256, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='data/DUTS')
    p.add_argument('--out', default='data/splits')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    root, out = Path(args.root)/'DUTS-TR', Path(args.out)
    pairs = paired_files(root/'DUTS-TR-Image', root/'DUTS-TR-Mask')
    if len(pairs) != 10553:
        raise ValueError(f'Expected 10553 DUTS-TR pairs, found {len(pairs)}')
    for index, (image, mask) in enumerate(pairs.values()):
        read_pair(image, mask)
        if (index+1) % 1000 == 0:
            print(f'Validated {index+1}/{len(pairs)} pairs', flush=True)
    train, val = split_ids(list(pairs), args.seed)
    out.mkdir(parents=True, exist_ok=True)
    for name, ids in [('train', train), ('val', val)]:
        target = out/f'{name}.txt'
        content = '\n'.join(ids)+'\n'
        if target.exists() and target.read_text(encoding='utf-8') != content:
            raise FileExistsError('Refusing to replace an existing different split')
        target.write_text(content, encoding='utf-8')
    write_json(out/'metadata.json', dict(seed=args.seed, train=len(train), validation=len(val),
               train_sha256=sha256(out/'train.txt'), val_sha256=sha256(out/'val.txt')))
    print(f'Train: {len(train)}, validation: {len(val)}')


if __name__ == '__main__':
    main()
