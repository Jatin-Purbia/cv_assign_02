"""Train an original FCN from scratch; configuration paths are relative to cwd."""
from src import runtime
runtime.configure()
import argparse
from src.trainer import train
from src.utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    train(read_json(args.config), args.out, args.device)


if __name__ == '__main__':
    main()
