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
    parser.add_argument('--resume', action='store_true', help='Restore a complete epoch checkpoint, including RNG state')
    parser.add_argument('--max-epochs', type=int, help='Run at most this many additional epochs, then checkpoint')
    args = parser.parse_args()
    train(read_json(args.config), args.out, args.device, resume=args.resume, max_epochs=args.max_epochs)


if __name__ == '__main__':
    main()
