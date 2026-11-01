"""Grading CLI: exactly two stdout lines on success, errors to stderr."""
from src import runtime
runtime.configure()
import argparse
import sys
from pathlib import Path
import cv2
from src.inference import Predictor
from src.preprocessing import read_image
from src.utils import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--weights', default=str(ROOT/'weights'/'final_model.pth'))
    args = parser.parse_args()
    if Path(args.out).suffix.lower() != '.png':
        raise ValueError('--out must be a PNG path')
    predictor = Predictor(args.weights)
    image = read_image(args.image)
    mask = predictor(image)  # Full-pipeline warmup and the output prediction.
    if not cv2.imwrite(args.out, mask*255):
        raise OSError(f'Could not save {args.out}')
    times = [predictor.timed(image)[1] for _ in range(20)]
    print(f'T_CPU_ms: {sum(times)/20:.2f}')
    print(f'S_disk_bytes: {predictor.size_bytes}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
