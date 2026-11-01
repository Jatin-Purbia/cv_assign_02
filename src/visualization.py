"""Plots of actual logs and observed masks; no illustrative metric values."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.dataset import paired_files, read_ids
from src.inference import Predictor
from src.metrics import MetricAccumulator
from src.preprocessing import read_image, read_mask
from src.utils import write_json, sha256


def plot_history(history: str | Path, output: str | Path) -> None:
    with open(history, encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('Training log is empty')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    epochs = [int(row['epoch']) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for field, label in [('train_loss', 'Training'), ('val_loss', 'Validation')]:
        axes[0, 0].plot(epochs, [float(r[field]) for r in rows], marker='o', label=label)
    axes[0, 0].set_ylabel('BCE + weighted soft Dice')
    axes[0, 0].legend()
    for axis, field, title in zip(axes.flat[1:], ['miou', 'bf1', 'q'], ['Global mIoU', 'Mean Boundary F1', 'Quality Q']):
        axis.plot(epochs, [float(r[field]) for r in rows], marker='o')
        axis.set_ylabel(title)
        axis.set_ylim(0, 1)
    for axis in axes.flat:
        axis.set_xlabel('Epoch')
        axis.grid(alpha=.2)
    fig.savefig(out/'training.png', dpi=180)
    plt.close(fig)


def overlays(weights: str | Path, images: str | Path, masks: str | Path,
             output: str | Path, manifest: str | Path | None = None) -> list[dict]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    pairs = paired_files(images, masks)
    ids = read_ids(manifest) if manifest else sorted(pairs)
    predictor, records = Predictor(weights), []
    for index, key in enumerate(ids):
        image = read_image(pairs[key][0])
        true, pred = read_mask(pairs[key][1]), predictor(image)
        rgb = image[:, :, ::-1].astype(np.float32)/255
        def overlay(mask):
            color = np.zeros_like(rgb)
            color[:, :, 1] = 1
            return np.where(mask[:, :, None].astype(bool), .55*rgb+.45*color, rgb)
        fig, axes = plt.subplots(1, 5, figsize=(12, 3.2), constrained_layout=True)
        for axis, data, title in zip(axes, [rgb, true, pred, overlay(true), overlay(pred)],
                                     ['Photograph', 'Ground truth', 'Prediction', 'True overlay', 'Predicted overlay']):
            axis.imshow(data, cmap='gray', vmin=0, vmax=1)
            axis.set_title(title, fontsize=14)
            axis.axis('off')
        fig.suptitle(key, fontsize=16)
        filename = f'sample_{index:04d}.png'
        fig.savefig(out/filename, dpi=160)
        plt.close(fig)
        metric = MetricAccumulator()
        metric.update(torch.from_numpy(pred), torch.from_numpy(true))
        records.append(dict(id=key, figure=filename, **metric.compute()))
    write_json(out/'index.json', records)
    write_json(out/'provenance.json', dict(checkpoint_sha256=sha256(weights),
               ids=ids, resolution=predictor.resolution, threshold=predictor.threshold))
    write_json(out/'failure_candidates.json', sorted(records, key=lambda r: r['q'])[:2])
    return records


def plot_comparison(table: str | Path, output: str | Path) -> None:
    with open(table, encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('Comparison has no observations')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for row in rows:
        axes[0].scatter(float(row['cpu_ms']), float(row['q']))
        axes[0].annotate(row['name'], (float(row['cpu_ms']), float(row['q'])), fontsize=7)
        axes[1].scatter(float(row['size_bytes'])/1048576, float(row['q']))
        axes[1].annotate(row['name'], (float(row['size_bytes'])/1048576, float(row['q'])), fontsize=7)
    axes[0].set(xlabel='CPU process time (ms)', ylabel='Validation Q')
    axes[1].set(xlabel='Saved size (MiB)', ylabel='Validation Q')
    scores = [float(r['ces']) for r in rows]
    best = max(range(len(rows)), key=lambda i: scores[i])
    axes[2].bar(range(len(rows)), scores, color=['#c84c30' if i == best else '#467698' for i in range(len(rows))])
    axes[2].set_xticks(range(len(rows)), [r['name'] for r in rows], rotation=65, ha='right', fontsize=7)
    axes[2].set_ylabel('Validation CES estimate (selected in red)')
    fig.savefig(out/'comparison.png', dpi=180)
    plt.close(fig)
    # Standalone report panels preserve readable typography at page width.
    for xkey, xlabel, filename in [('cpu_ms', 'CPU process time (ms)', 'q_vs_time.png'),
                                  ('size_bytes', 'Saved size (MiB)', 'size_vs_q.png')]:
        fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
        seen = set()
        for i, row in enumerate(rows):
            x = float(row[xkey])/(1048576 if xkey == 'size_bytes' else 1)
            y = float(row['q'])
            group = 'Depthwise BCE' if 'depthwise' in row['name'] else ('Tiny BCE+Dice' if 'dice' in row['name'] else 'Tiny BCE')
            color, marker = {'Depthwise BCE': ('#26745e', 's'), 'Tiny BCE+Dice': ('#ac7924', '^'),
                             'Tiny BCE': ('#467698', 'o')}[group]
            label = 'Selected' if i == best else (group if group not in seen else None)
            if i != best:
                seen.add(group)
            ax.scatter(x, y, c='#c84c30' if i == best else color,
                       marker='*' if i == best else marker, s=150 if i == best else 40, label=label)
        ax.set(xlabel=xlabel, ylabel='Validation Q', title='Selected candidate marked with red star')
        if xkey == 'cpu_ms':
            ax.axvline(50, color='#8b3232', linestyle='--', linewidth=1, label='50 ms hard limit')
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
        fig.savefig(out/filename, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, min(8, max(3.5, .4*len(rows)+1))), constrained_layout=True)
    ax.barh(range(len(rows)), scores, color=['#c84c30' if i == best else '#467698' for i in range(len(rows))])
    ax.set_yticks(range(len(rows)), [f"{i+1}: {r['name']}" for i, r in enumerate(rows)], fontsize=9)
    ax.set(xlabel='Validation CES estimate', title='Validation-only selection (red)')
    ax.invert_yaxis()
    fig.savefig(out/'ces_comparison.png', dpi=180)
    plt.close(fig)
