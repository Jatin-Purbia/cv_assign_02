from src import runtime
runtime.configure()
import argparse
from pathlib import Path
from src.dataset import paired_files
from src.preprocessing import read_pair
from src.captured import verify_captured


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--split', choices=['TR', 'TE', 'captured'], required=True)
    args = p.parse_args()
    if args.split == 'captured':
        print(verify_captured(args.root))
        return
    root = Path(args.root)/f'DUTS-{args.split}'
    pairs = paired_files(root/f'DUTS-{args.split}-Image', root/f'DUTS-{args.split}-Mask')
    expected = 10553 if args.split == 'TR' else 5019
    if len(pairs) != expected:
        raise ValueError(f'Expected {expected} pairs, got {len(pairs)}')
    for image, mask in pairs.values():
        read_pair(image, mask)
    print(f'Validated {len(pairs)} decodable matched pairs')


if __name__ == '__main__':
    main()
