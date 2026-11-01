"""Generate masks, a visual comparison, and provenance for a folder of photos."""
from src import runtime
runtime.configure()
import argparse
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.dataset import indexed_files
from src.deployment import active_weights
from src.optimized_inference import ConfiguredPredictor as Predictor
from src.preprocessing import read_image
from src.utils import sha256, write_json


def predict_folder(images: Path, output: Path, weights: Path, compare_weights: Path | None = None) -> list:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty output folder to preserve earlier predictions')
    files = indexed_files(images)
    if not files:
        raise ValueError('No supported photos found')
    output.mkdir(parents=True, exist_ok=True)
    (output/'masks').mkdir()
    predictor = Predictor(weights)
    baseline = Predictor(compare_weights) if compare_weights else None
    columns = 4 if baseline else 3
    records, montages = [], []
    page_size = 10
    def overlay(rgb, mask):
        green = np.zeros_like(rgb)
        green[:, :, 1] = 1
        return np.where(mask[:, :, None].astype(bool), .6*rgb+.4*green, rgb)
    for row, (key, path) in enumerate(files.items()):
        if row % page_size == 0:
            rows = min(page_size, len(files)-row)
            fig, axes = plt.subplots(rows, columns, figsize=(3*columns, 2.8*rows),
                                     squeeze=False, constrained_layout=True)
        photo = read_image(path)
        mask = predictor(photo)
        destination = output/'masks'/f'{key}.png'
        if not cv2.imwrite(str(destination), mask*255):
            raise OSError(f'Cannot write mask: {destination}')
        rgb = photo[:, :, ::-1].astype(np.float32)/255
        panels = [(rgb, key)]
        if baseline:
            panels.append((overlay(rgb, baseline(photo)), 'Baseline'))
        panels += [(overlay(rgb, mask), 'Selected checkpoint'), (mask, 'Selected binary mask')]
        for axis, (data, title) in zip(axes[row % page_size], panels):
            axis.imshow(data, cmap='gray', vmin=0, vmax=1)
            axis.set_title(title, fontsize=10)
            axis.axis('off')
        area = float(mask.mean())
        flags = ['very_little_foreground'] if area < .01 else ['most_image_foreground'] if area > .8 else []
        records.append(dict(id=key, input_sha256=sha256(path), mask=destination.relative_to(output).as_posix(),
                            foreground_fraction=area, review_flags=flags))
        if row % page_size == page_size-1 or row == len(files)-1:
            filename = 'comparison.png' if not montages else f'comparison_{len(montages)+1:03d}.png'
            fig.savefig(output/filename, dpi=130)
            plt.close(fig)
            montages.append(filename)
    write_json(output/'predictions.json', dict(checkpoint_sha256=sha256(weights),
               checkpoint=str(weights), resolution=predictor.resolution, threshold=predictor.threshold,
               memory_format=predictor.memory_format,
               comparison_sha256=sha256(compare_weights) if compare_weights else None,
               note='No ground truth; review flags are area heuristics, not calibrated confidence',
               images=records, montages=montages))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', default='captured/original')
    parser.add_argument('--out', default='captured/prediction_runs')
    parser.add_argument('--weights')
    parser.add_argument('--compare-weights', default='weights/final_model.pth')
    args = parser.parse_args()
    records = predict_folder(Path(args.images), Path(args.out),
                             Path(args.weights) if args.weights else active_weights(),
                             Path(args.compare_weights) if args.compare_weights else None)
    print(f'Wrote {len(records)} masks and comparison.png to {args.out}')


if __name__ == '__main__':
    main()
