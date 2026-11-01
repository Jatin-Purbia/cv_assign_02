"""Reproducible photometric checks on validation photos, never captured/test data."""
from pathlib import Path
import numpy as np
import torch
from src.dataset import paired_files, read_ids
from src.optimized_inference import ConfiguredPredictor as Predictor
from src.metrics import MetricAccumulator
from src.preprocessing import read_pair
from src.utils import ROOT, sha256


def photometric(image: np.ndarray, condition: str) -> np.ndarray:
    x = image.astype(np.float32)
    if condition == 'dark':
        x *= .6
    elif condition == 'bright':
        x *= 1.3
    elif condition == 'low_contrast':
        mean = x.mean(axis=(0, 1), keepdims=True)
        x = mean + .6*(x-mean)
    else:
        raise ValueError('Unknown stress condition')
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def stress_validation(weights: str | Path, count: int = 128) -> dict:
    if count < 1:
        raise ValueError('Stress sample count must be positive')
    manifest = ROOT/'data/splits/val.txt'
    all_ids = read_ids(manifest)
    # Spread deterministic samples across the saved, sorted validation manifest.
    indices = np.linspace(0, len(all_ids)-1, min(count, len(all_ids)), dtype=int)
    ids = [all_ids[int(i)] for i in indices]
    root = ROOT/'data/DUTS/DUTS-TR'
    pairs = paired_files(root/'DUTS-TR-Image', root/'DUTS-TR-Mask')
    predictor = Predictor(weights)
    accumulators = {k: MetricAccumulator() for k in ('dark', 'bright', 'low_contrast')}
    for key in ids:
        image, target = read_pair(*pairs[key])
        for condition, metric in accumulators.items():
            mask = predictor(photometric(image, condition))
            metric.update(torch.from_numpy(mask), torch.from_numpy(target))
    conditions = {k: v.compute() for k, v in accumulators.items()}
    return dict(split='validation-stress', manifest_sha256=sha256(manifest),
                checkpoint_sha256=sha256(weights), ids=ids, conditions=conditions,
                mean_q=sum(v['q'] for v in conditions.values())/len(conditions))


def stress_gates(baseline: dict, candidate: dict) -> dict:
    if baseline['ids'] != candidate['ids'] or baseline['manifest_sha256'] != candidate['manifest_sha256']:
        raise ValueError('Stress observations must use the same validation images')
    differences = {k: candidate['conditions'][k]['q']-v['q'] for k, v in baseline['conditions'].items()}
    checks = dict(mean_nonregression=candidate['mean_q'] >= baseline['mean_q'],
                  condition_nonregression=all(v >= -.01 for v in differences.values()))
    return dict(passed=all(checks.values()), checks=checks, q_gains=differences,
                mean_q_gain=candidate['mean_q']-baseline['mean_q'],
                policy=dict(min_mean_q_gain=0, max_condition_q_drop=.01))
