"""Versioned, validation-qualified model selection without changing the baseline."""
from pathlib import Path
from src.inference import verify_frozen
from src.utils import ROOT, read_json, environment


def active_weights() -> Path:
    manifest = ROOT/'weights/active_model.json'
    if not manifest.exists():
        return ROOT/'weights/final_model.pth'
    release = read_json(manifest)
    if release.get('format_version') != 1 or release.get('task') != 'automatic-salient-object':
        raise ValueError('Unsupported active model manifest')
    if not release.get('gates', {}).get('passed'):
        raise ValueError('Active model has not passed validation qualification')
    if not release.get('stress_gates', {}).get('passed'):
        raise ValueError('Active model has not passed lighting robustness checks')
    validated = release.get('validated_environment')
    if validated:
        current = environment()
        differences = [key for key in ('torch', 'numpy', 'opencv')
                       if current[key] != validated[key]]
        if current['python'].split('.')[:2] != validated['python'].split('.')[:2]:
            differences.append('python')
        if differences:
            raise ValueError(f"Active release requires its validated runtime: Python {validated['python']}, "
                             f"PyTorch {validated['torch']}, NumPy {validated['numpy']}, "
                             f"OpenCV {validated['opencv']}. Mismatches: {', '.join(differences)}. "
                             'Use the documented Python runtime, or --weights for an explicit experimental benchmark.')
    path = (ROOT/release['weights']).resolve()
    if not path.is_relative_to((ROOT/'weights/releases').resolve()):
        raise ValueError('Active model must be under weights/releases')
    frozen = verify_frozen(path)
    if frozen['sha256'] != release['sha256']:
        raise ValueError('Active manifest and release checkpoint disagree')
    return path


def release_gates(baseline: dict, candidate: dict) -> dict:
    """Engineering acceptance criteria, not a claim of real-world reliability."""
    if baseline['split'] != 'validation' or candidate['split'] != 'validation':
        raise ValueError('Release qualification requires validation evidence')
    if baseline['manifest_sha256'] != candidate['manifest_sha256']:
        raise ValueError('Baseline and candidate must use the same validation manifest')
    before = {row['id']: row for row in baseline['per_image']}
    after = {row['id']: row for row in candidate['per_image']}
    if not before or before.keys() != after.keys() or len(after) != candidate['images']:
        raise ValueError('Qualification observations do not cover identical unique images')
    times = sorted(row['cpu_ms'] for row in candidate['per_image'])
    p95 = times[max(0, (95*len(times)+99)//100-1)]
    regression_rate = sum(after[k]['q'] < before[k]['q']-.05 for k in before)/len(before)
    checks = dict(quality_gain=candidate['q'] >= baseline['q']+.005,
                  boundary_nonregression=candidate['bf1'] >= baseline['bf1'],
                  region_nonregression=candidate['miou'] >= baseline['miou']-.002,
                  mean_cpu_margin=candidate['cpu_ms'] <= 40,
                  p95_cpu_budget=p95 <= 50,
                  limited_large_regressions=regression_rate <= .25)
    return dict(passed=all(checks.values()), checks=checks,
                q_gain=candidate['q']-baseline['q'], bf1_gain=candidate['bf1']-baseline['bf1'],
                p95_cpu_ms=p95, large_regression_fraction=regression_rate,
                policy=dict(min_q_gain=.005, max_miou_drop=.002, max_mean_cpu_ms=40,
                            max_p95_cpu_ms=50, max_large_regression_fraction=.25,
                            large_regression_q_drop=.05))
