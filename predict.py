"""Grading CLI: exactly two stdout lines on success, errors to stderr."""
from src import runtime
runtime.configure()
import argparse
import sys
import uuid
from pathlib import Path
import cv2
from src.optimized_inference import ConfiguredPredictor as Predictor
from src.preprocessing import read_image
from src.deployment import active_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--weights', help='Explicit checkpoint; otherwise use the qualified active model or baseline')
    args = parser.parse_args()
    if Path(args.out).suffix.lower() != '.png':
        raise ValueError('--out must be a PNG path')
    if Path(args.image).resolve() == Path(args.out).resolve():
        raise ValueError('Input and output paths must differ')
    predictor = Predictor(args.weights or active_weights())
    image = read_image(args.image)
    mask = predictor(image)  # Full-pipeline warmup and the output prediction.
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem+'.'+uuid.uuid4().hex+'.tmp.png')
    try:
        if not cv2.imwrite(str(temporary), mask*255):
            raise OSError(f'Could not save {args.out}')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    times = [predictor.timed(image)[1] for _ in range(20)]
    print(f'T_CPU_ms: {sum(times)/20:.2f}')
    print(f'S_disk_bytes: {predictor.size_bytes}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
