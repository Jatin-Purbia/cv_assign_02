"""Draw the report's evidence plots from saved result files (Matplotlib + NumPy only)."""
from src import runtime
runtime.configure()
import argparse
import csv
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.utils import ROOT

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e3e2de'
FINAL_SIZE = 1787043
ACTIVE_DUTS = 'results/quality/duts_test_metrics.json'
ACTIVE_VAL = 'results/quality/unet16_dice_aug_recovered_r128_t0.5_channels_last.validation.json'
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.labelcolor': MUTED,
    'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.edgecolor': GRID,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 100, 'savefig.dpi': 200,
    'legend.frameon': False})


def load(path: str) -> dict:
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def ces(q, ms, size=FINAL_SIZE):
    return q*(1-0.20*ms/50-0.10*min(1, size/4194304)) if ms <= 50 else 0.0


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out/name, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', out/name)


def distributions(out: Path) -> None:
    duts = load(ACTIVE_DUTS)['per_image']
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))
    for ax, key, label in zip(axes, ('miou', 'bf1', 'q'), ('mIoU (per image)', 'BF1 (per image)', 'Q (per image)')):
        v = np.array([r[key] for r in duts])
        ax.hist(v, bins=40, color=BLUE, edgecolor='white', linewidth=0.4)
        ax.axvline(v.mean(), color=ORANGE, lw=1.6)
        ax.text(0.03, 0.95, f'mean {v.mean():.3f}', transform=ax.transAxes, ha='left', va='top', fontsize=8)
        ax.set_xlabel(label)
    axes[0].set_ylabel('images')
    fig.suptitle('DUTS-TE per-image score distributions (5,019 images)', fontweight='bold')
    fig.tight_layout()
    save(fig, out, 'fig_distributions.png')


def hexbin(out: Path) -> None:
    duts = load(ACTIVE_DUTS)['per_image']
    x = np.array([r['miou'] for r in duts])
    y = np.array([r['bf1'] for r in duts])
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    hb = ax.hexbin(x, y, gridsize=32, cmap='Blues', mincnt=1)
    fig.colorbar(hb, ax=ax, label='images per cell')
    ax.set_xlabel('mIoU (per image)')
    ax.set_ylabel('BF1 (per image)')
    ax.set_title('Overlap vs boundary quality on DUTS-TE')
    fig.tight_layout()
    save(fig, out, 'fig_hexbin.png')


def resolution(out: Path) -> None:
    base = {96: load('results/quality/baseline_validation.json'),
            112: load('results/quality/baseline_r112_t0.5_channels_last.validation.json'),
            128: load('results/quality/baseline_r128_t0.5_channels_last.validation.json')}
    tuned = {r: load(f'results/quality/unet16_dice_aug_recovered_r{r}_t0.5_channels_last.validation.json')
             for r in (96, 112, 128, 144)}
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    for ax, key, label in zip(axes, ('q', 'cpu_ms', 'ces'), ('Q', 'CPU time (ms)', 'CES')):
        ax.plot(list(base), [base[r][key] for r in base], '-o', color=BLUE, lw=2, ms=5, label='Stage 1')
        ax.plot(list(tuned), [tuned[r][key] for r in tuned], '-o', color=ORANGE, lw=2, ms=5, label='Fine-tuned')
        ax.set_xlabel('internal resolution r (px)')
        ax.set_ylabel(label)
        ax.set_xticks([96, 112, 128, 144])
    axes[0].legend(loc='lower right')
    fig.suptitle('Validation effect of the internal resolution', fontweight='bold')
    fig.tight_layout()
    save(fig, out, 'fig_resolution.png')


def tradeoff(out: Path) -> None:
    rows = list(csv.DictReader((ROOT/'results/model_comparison.csv').open(encoding='utf-8')))
    families: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        m = re.fullmatch(r'(.+)_infer\d+', r['name'])
        if m:
            families.setdefault(m.group(1), []).append((float(r['cpu_ms']), float(r['q'])))
    names = {'tiny4_bce_128': 'Tiny FCN, BCE', 'tiny4_dice_128': 'Tiny FCN, BCE+Dice',
             'depthwise8_bce_128': 'Depthwise FCN', 'unet16_bce_128': 'U-Net (stage 1)'}
    colours = [BLUE, '#eda100', AQUA, '#6250d6']
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ts = np.linspace(2, 60, 120)
    for level in (0.40, 0.50, 0.60, 0.70):
        qs = level/(1-0.20*ts/50-0.10*FINAL_SIZE/4194304)
        keep = qs <= 0.95
        ax.plot(ts[keep], qs[keep], color=GRID, lw=1.2, zorder=1)
        ax.text(ts[keep][-1], qs[keep][-1], f' CES {level:.2f}', color=MUTED, fontsize=7, va='center')
    for (key, label), colour in zip(names.items(), colours):
        pts = sorted(families[key])
        ax.plot(*zip(*pts), '-o', color=colour, lw=1.8, ms=4.5, label=label, zorder=3)
    tuned = [load(f'results/quality/unet16_dice_aug_recovered_r{r}_t0.5_channels_last.validation.json')
             for r in (96, 112, 128, 144)]
    ax.plot([t['cpu_ms'] for t in tuned], [t['q'] for t in tuned], '-o', color=ORANGE, lw=2.4, ms=6,
            label='U-Net (fine-tuned)', zorder=4)
    ax.scatter([tuned[2]['cpu_ms']], [tuned[2]['q']], s=150, facecolor='none', edgecolor=INK, lw=1.4, zorder=5)
    ax.annotate('selected release', (tuned[2]['cpu_ms'], tuned[2]['q']), textcoords='offset points',
                xytext=(10, 12), fontsize=8)
    ax.axvline(50, color=MUTED, ls='--', lw=1)
    ax.text(49, 0.50, '50 ms limit', ha='right', color=MUTED, fontsize=8)
    ax.set_xscale('log')
    ax.set_xlim(3, 65)
    ax.set_xticks([4, 6, 10, 20, 40], ['4', '6', '10', '20', '40'])
    ax.set_xlabel('mean CPU time per image (ms, log scale; points are inference resolutions)')
    ax.set_ylabel('validation Q')
    ax.set_title('Quality against cost across model families')
    ax.legend(loc='center left', bbox_to_anchor=(0.0, 0.62), fontsize=8)
    fig.tight_layout()
    save(fig, out, 'fig_tradeoff.png')


def ces_curves(out: Path) -> None:
    ts = np.linspace(0, 60, 241)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for q, colour in zip((0.5, 0.6, 0.7, 0.8), ('#86b6ef', '#5598e7', BLUE, '#184f95')):
        ax.plot(ts, [ces(q, t) for t in ts], color=colour, lw=2, label=f'Q = {q:.1f}')
    marks = [('Validation', 24.32, 0.7899), ('DUTS-TE', 17.14, 0.6333), ('Captured', 14.06, 0.5329)]
    for name, ms, q in marks:
        ax.scatter(ms, ces(q, ms), s=55, color=ORANGE, edgecolor='white', linewidth=1.2, zorder=4)
        ax.annotate(name, (ms, ces(q, ms)), textcoords='offset points', xytext=(6, 6), fontsize=8)
    ax.axvline(50, color=MUTED, ls='--', lw=1)
    ax.text(49.5, 0.1, 'budget limit', ha='right', color=MUTED, fontsize=8)
    ax.set_xlabel('CPU time per image (ms)')
    ax.set_ylabel('CES')
    ax.set_title('How CES depends on speed (file size 1.79 MB)')
    ax.legend(loc='lower left')
    fig.tight_layout()
    save(fig, out, 'fig_ces_curves.png')


def thresholds(out: Path) -> None:
    taus = [0.4, 0.45, 0.5, 0.55, 0.6]
    sweep = [load(f'experiments/final/unet16_bce_128/threshold_r96_{t}.validation.json') for t in taus]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for key, label, colour in (('q', 'Q', BLUE), ('miou', 'mIoU', AQUA), ('bf1', 'BF1', ORANGE)):
        axes[0].plot(taus, [s[key] for s in sweep], '-o', color=colour, lw=2, ms=5, label=label)
    axes[0].set_xlabel('decision threshold')
    axes[0].set_ylabel('validation score')
    axes[0].set_title('Stage-1 threshold sweep (r = 96)')
    axes[0].legend(loc='center right')
    res = [96, 112, 128, 144]
    for tau, colour in (('0.45', BLUE), ('0.5', ORANGE)):
        vals = [load(f'results/quality/unet16_dice_aug_recovered_r{r}_t{tau}_channels_last.validation.json')['q']
                for r in res]
        axes[1].plot(res, vals, '-o', color=colour, lw=2, ms=6, label=f'threshold {tau}')
    axes[1].set_xticks(res)
    axes[1].set_xlabel('internal resolution r (px)')
    axes[1].set_ylabel('validation Q')
    axes[1].set_title('Fine-tuned: threshold 0.45 vs 0.5')
    axes[1].legend(loc='lower center')
    fig.tight_layout()
    save(fig, out, 'fig_threshold.png')


def confusion(out: Path) -> None:
    sets = [('Validation', ACTIVE_VAL), ('DUTS-TE', ACTIVE_DUTS),
            ('Captured photos', 'results/captured_reference_metrics.json')]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    for ax, (name, path) in zip(axes, sets):
        c = np.array(load(path)['confusion_matrix'], dtype=float)
        n = c/c.sum(axis=1, keepdims=True)
        ax.imshow(n, cmap='Blues', vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{100*n[i, j]:.1f}%', ha='center', va='center',
                        color='white' if n[i, j] > 0.5 else INK, fontsize=10)
        ax.set_xticks([0, 1], ['background', 'object'])
        ax.set_yticks([0, 1], ['background', 'object'])
        ax.set_xlabel('predicted')
        ax.set_title(name)
        ax.grid(False)
    axes[0].set_ylabel('true')
    fig.suptitle('Row-normalised pixel confusion matrices', fontweight='bold')
    fig.tight_layout()
    save(fig, out, 'fig_confusion.png')


def captured(out: Path) -> None:
    active = {r['id']: r for r in load('results/captured_reference_metrics.json')['per_image']}
    base = {r['id']: r for r in load('results/captured_reference_metrics_baseline.json')['per_image']}
    ids = sorted(active, key=lambda k: active[k]['q'])
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8), gridspec_kw={'width_ratios': [1.15, 1]})
    y = np.arange(len(ids))
    axes[0].barh(y+0.2, [base[i]['q'] for i in ids], 0.38, color=BLUE, label='Stage 1')
    axes[0].barh(y-0.2, [active[i]['q'] for i in ids], 0.38, color=ORANGE, label='Fine-tuned')
    axes[0].set_yticks(y, [i.replace('_', ' ') for i in ids])
    axes[0].set_xlabel('Q on the photo')
    axes[0].set_title('Per-photo Q')
    axes[0].legend(loc='lower right')
    axes[1].scatter([active[i]['miou'] for i in ids], [active[i]['bf1'] for i in ids],
                    s=55, color=ORANGE, edgecolor='white', linewidth=1.2, zorder=3)
    for i in ids:
        axes[1].annotate(i.replace('_', ' '), (active[i]['miou'], active[i]['bf1']),
                         textcoords='offset points', xytext=(5, 3), fontsize=7)
    axes[1].set_xlabel('mIoU')
    axes[1].set_ylabel('BF1')
    axes[1].set_title('Overlap vs boundary')
    fig.tight_layout()
    save(fig, out, 'fig_captured.png')


def stress(out: Path) -> None:
    base = load('results/quality/baseline_stress.json')['conditions']
    tuned = load('results/quality/unet16_dice_aug_recovered_r128_t0.5_channels_last.stress.json')['conditions']
    names = ['dark', 'bright', 'low_contrast']
    xs = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(xs, [base[n]['q'] for n in names], 'o', color=BLUE, ms=8, label='Stage 1')
    ax.plot(xs, [tuned[n]['q'] for n in names], 'o', color=ORANGE, ms=8, label='Fine-tuned')
    for x, n in zip(xs, names):
        ax.plot([x, x], [base[n]['q'], tuned[n]['q']], color=GRID, lw=2, zorder=0)
    ax.set_xticks(xs, ['dark', 'bright', 'low contrast'])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel('validation-subset Q')
    ax.set_title('Robustness to lighting changes (128 images)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    save(fig, out, 'fig_stress.png')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='report/figures')
    args = p.parse_args()
    out = ROOT/args.out
    out.mkdir(parents=True, exist_ok=True)
    for draw in (distributions, hexbin, resolution, tradeoff, ces_curves, thresholds, confusion, captured, stress):
        draw(out)


if __name__ == '__main__':
    main()
