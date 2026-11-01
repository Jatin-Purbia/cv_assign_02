"""Paired image bootstrap of global mIoU plus mean Boundary F1."""
import numpy as np


def paired_quality_interval(baseline: dict, candidate: dict, draws: int = 2000, seed: int = 42) -> dict:
    if draws < 100:
        raise ValueError('At least 100 bootstrap draws are required')
    before = {r['id']: r for r in baseline['per_image']}
    after = {r['id']: r for r in candidate['per_image']}
    if not before or before.keys() != after.keys():
        raise ValueError('Paired bootstrap requires identical nonempty sample IDs')
    ids = sorted(before)
    counts = [np.array([records[k]['confusion_matrix'] for k in ids], dtype=np.float64).reshape(-1, 4)
              for records in (before, after)]
    boundaries = [np.array([records[k]['bf1'] for k in ids]) for records in (before, after)]
    if any(not np.isfinite(c).all() or (c < 0).any() for c in counts):
        raise ValueError('Invalid per-image confusion counts')
    if not np.array_equal(counts[0][:, :2].sum(1), counts[1][:, :2].sum(1)) or not np.array_equal(
            counts[0][:, 2:].sum(1), counts[1][:, 2:].sum(1)):
        raise ValueError('Baseline and candidate target class counts differ')
    generator = np.random.default_rng(seed)
    differences = []
    for start in range(0, draws, 128):
        indices = generator.integers(0, len(ids), size=(min(128, draws-start), len(ids)))
        scores = []
        for matrix, bf1 in zip(counts, boundaries):
            tn, fp, fn, tp = matrix[indices].sum(1).T
            errors = fp+fn
            fg = np.divide(tp, tp+errors, out=np.ones_like(tp), where=tp+errors > 0)
            bg = np.divide(tn, tn+errors, out=np.ones_like(tn), where=tn+errors > 0)
            scores.append(.7*(fg+bg)/2 + .3*bf1[indices].mean(1))
        differences.extend((scores[1]-scores[0]).tolist())
    low, high = np.quantile(differences, [.025, .975])
    return dict(passed=bool(low > 0), q_gain_95_interval=[float(low), float(high)],
                draws=draws, seed=seed,
                method='Paired image bootstrap; global confusion IoU and mean per-image BF1',
                limitation='Validation-sampling uncertainty only; not deployment-domain reliability')
