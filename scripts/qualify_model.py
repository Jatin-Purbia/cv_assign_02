"""Compare candidates on DUTS-TR validation, then optionally activate a versioned release.

Never reads DUTS-TE or captured photos. Run after training finishes so CPU timings
are not contaminated by a concurrent training process.
"""
from src import runtime
runtime.configure()
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
from src.deployment import release_gates
from src.evaluation import evaluate
from src.optimized_inference import ConfiguredPredictor as Predictor, export_configured_weights
from src.robustness import stress_validation, stress_gates
from src.statistics import paired_quality_interval
from src.utils import ROOT, read_json, write_json, sha256


def qualify(runs: list[str], output: Path, activate: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    data = ROOT/'data/DUTS/DUTS-TR'
    manifest = ROOT/'data/splits/val.txt'
    baseline_path = ROOT/'weights/final_model.pth'
    baseline = evaluate(baseline_path, data/'DUTS-TR-Image', data/'DUTS-TR-Mask',
                        output/'baseline_validation.json', 'validation', manifest)
    observations = []
    sources = [('baseline', baseline_path, None)]
    for name in runs:
        run_path = Path(name)
        run = read_json(run_path/'run.json')
        summary = read_json(run_path/'summary.json')
        if run['config'].get('synthetic_smoke') or not summary.get('finished', True):
            raise ValueError('Only completed real training runs may be qualified')
        if run['val_manifest_sha256'] != sha256(manifest):
            raise ValueError('Training validation manifest changed')
        sources.append((run_path.name, run_path/'best.pth', run))
    # The old checkpoint at higher resolutions is a control, so retraining must
    # beat simple inference changes before it earns a release.
    for label, source, run in sources:
        predictor = Predictor(source)
        resolutions = (112, 128) if label == 'baseline' else (96, 112, 128, 144)
        thresholds = (.5,) if label == 'baseline' else (.45, .5)
        formats = ('contiguous', 'channels_last') if label == 'baseline' else ('channels_last',)
        for resolution in resolutions:
            for threshold, memory_format in [(t, f) for t in thresholds for f in formats]:
                name = f'{label}_r{resolution}_t{threshold:g}_{memory_format}'
                weights = output/f'{name}.pth'
                export_configured_weights(weights, predictor.model, predictor.config, resolution, threshold, memory_format)
                validation_path = output/f'{name}.validation.json'
                value = evaluate(weights, data/'DUTS-TR-Image', data/'DUTS-TR-Mask',
                                 validation_path, 'validation', manifest)
                gates = release_gates(baseline, value)
                interval = paired_quality_interval(baseline, value)
                gates['checks']['positive_paired_interval'] = interval['passed']
                gates['passed'] = all(gates['checks'].values())
                observations.append(dict(name=name, weights=weights.relative_to(ROOT).as_posix(),
                    validation_path=validation_path.relative_to(ROOT).as_posix(),
                    run=run, memory_format=memory_format, gates=gates, paired_quality_interval=interval,
                    metrics={k: value[k] for k in ('miou', 'bf1', 'q', 'cpu_ms', 'size_bytes', 'ces')}))
                print(f"{name}: Q={value['q']:.5f}, BF1={value['bf1']:.5f}, "
                      f"CPU={value['cpu_ms']:.2f}ms, gates={gates['passed']}", flush=True)
    passing = [v for v in observations if v['gates']['passed']]
    baseline_stress = stress_validation(baseline_path)
    write_json(output/'baseline_stress.json', baseline_stress)
    selected = None
    for candidate in sorted(passing, key=lambda v: (v['metrics']['q'], -v['metrics']['cpu_ms']), reverse=True):
        stress = stress_validation(ROOT/candidate['weights'])
        write_json(output/f"{candidate['name']}.stress.json", stress)
        candidate['stress_metrics'] = stress
        candidate['stress_gates'] = stress_gates(baseline_stress, stress)
        if candidate['stress_gates']['passed']:
            selected = candidate
            break
    report = dict(baseline_sha256=sha256(baseline_path),
                  baseline={k: baseline[k] for k in ('miou', 'bf1', 'q', 'cpu_ms', 'size_bytes', 'ces', 'images', 'manifest_sha256')},
                  baseline_stress=baseline_stress,
                  qualification_source_sha256={f: sha256(ROOT/f) for f in
                      ('src/deployment.py', 'src/robustness.py', 'src/statistics.py', 'src/evaluation.py',
                       'src/optimized_inference.py', 'scripts/qualify_model.py')},
                  scope='DUTS-TR validation only; quality priority within CPU budget',
                  evaluated_utc=datetime.now(timezone.utc).isoformat(), observations=observations,
                  selected=selected, activated=False)
    if activate and selected:
        source = ROOT/selected['weights']
        digest = sha256(source)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        release_dir = ROOT/'weights/releases'/f'{digest[:16]}_{stamp}'
        release_dir.mkdir(parents=True, exist_ok=False)
        destination = release_dir/'model.pth'
        if destination.exists() and sha256(destination) != digest:
            raise ValueError('Existing release hash mismatch')
        if not destination.exists():
            temporary = destination.with_suffix('.pth.tmp')
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        selected_validation = read_json(ROOT/selected['validation_path'])
        frozen = dict(sha256=digest,
                      source_sha256={f: sha256(ROOT/f) for f in
                          ('src/inference.py', 'src/preprocessing.py', 'src/models.py', 'src/runtime.py',
                           'src/optimized_inference.py')},
                      frozen_utc=datetime.now(timezone.utc).isoformat(),
                      validation=selected_validation,
                      selection={k: v for k, v in selected.items() if k != 'run'},
                      run=selected['run'], statement='Selected using DUTS-TR validation only')
        write_json(destination.with_suffix('.freeze.json'), frozen)
        active = dict(format_version=1, task='automatic-salient-object', sha256=digest,
                      weights=destination.relative_to(ROOT).as_posix(), gates=selected['gates'],
                      stress_gates=selected['stress_gates'],
                      validated_environment=selected_validation['environment'],
                      qualification=(release_dir/'qualification.json').relative_to(ROOT).as_posix())
        # Complete evidence is committed before the active pointer switches.
        report['activated'] = True
        report['release_weights'] = destination.relative_to(ROOT).as_posix()
        write_json(release_dir/'qualification.json', report)
        write_json(output/'qualification.json', report)
        write_json(ROOT/'weights/active_model.json', active)
    else:
        write_json(output/'qualification.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='*', default=[],
                        help='Completed fine-tuning runs; omit to qualify inference-resolution changes only')
    parser.add_argument('--out', default='results/quality')
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--wait-hours', type=float, default=0,
                        help='Wait for every requested training summary before qualification (0: no wait)')
    args = parser.parse_args()
    if args.wait_hours < 0:
        parser.error('--wait-hours must be nonnegative')
    deadline = time.monotonic() + args.wait_hours*3600
    if args.wait_hours:
        print('Waiting for completed training summaries before qualification.', flush=True)
        while not all((Path(run)/'summary.json').exists() and
                      read_json(Path(run)/'summary.json').get('finished', False)
                      for run in args.runs):
            if time.monotonic() >= deadline:
                raise TimeoutError('Training did not finish before the qualification deadline')
            time.sleep(10)
    report = qualify(args.runs, Path(args.out).resolve(), args.activate)
    print('Selected:', report['selected']['name'] if report['selected'] else 'baseline retained')


if __name__ == '__main__':
    main()
