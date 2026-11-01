"""Source compliance inventory and strict submission audit; never creates fake prerequisites."""
from src import runtime
runtime.configure()
import argparse
import ast
from pathlib import Path
import sys
from src.utils import ROOT, write_json, read_json, sha256

ALLOWED = ({'torch', 'numpy', 'cv2', 'matplotlib', 'src', 'scripts', 'tests'}
           | sys.stdlib_module_names | {p.stem for p in ROOT.glob('*.py')})
BAD_PARTS = {'torchvision', 'sklearn', 'skimage', 'scipy', 'networkx', 'maxflow',
             'transformers', 'segmentation_models_pytorch', 'onnxruntime', 'openvino'}
CV_ALLOWED = {'imread', 'imwrite', 'resize', 'cvtColor', 'filter2D', 'setNumThreads', 'getNumThreads'}


def audit_source() -> dict:
    imports, issues = [], []
    files = list(ROOT.glob('*.py'))
    for folder in ('src', 'scripts', 'tests'):
        files.extend((ROOT/folder).rglob('*.py'))
    for path in sorted(files):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                imports.append(dict(file=str(path.relative_to(ROOT)), line=node.lineno, module=module))
                if module.split('.')[0] not in ALLOWED or set(module.split('.')) & BAD_PARTS:
                    issues.append(f'{path.name}:{node.lineno}: prohibited import {module}')
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'cv2' and node.func.attr not in CV_ALLOWED:
                    issues.append(f'{path.name}:{node.lineno}: unreviewed cv2.{node.func.attr}')
    return dict(imports=imports, issues=issues,
                scope='AST imports and direct cv2 calls; supplemented by manual code review')


def submission_status() -> dict:
    required = ['weights/final_model.pth', 'weights/final_model.freeze.json',
                'data/splits/train.txt', 'data/splits/val.txt', 'results/validation_metrics.json',
                'results/duts_test_metrics.json', 'results/captured_metrics.json',
                'results/model_comparison.csv', 'captured/metadata.json', 'report/main.pdf',
                'report/failure_analysis.tex', 'results/captured_figures/index.json',
                'results/captured_figures/provenance.json', 'results/tests.json']
    missing = [name for name in required if not (ROOT/name).is_file()]
    issues = []
    for name, split in [('validation_metrics', 'validation'), ('duts_test_metrics', 'duts-test'),
                        ('captured_metrics', 'captured')]:
        path = ROOT/'results'/f'{name}.json'
        if path.exists():
            value = read_json(path)
            if value.get('split') != split:
                issues.append(f'Incorrect split for {name}')
            if split == 'duts-test' and (value.get('images') != 5019 or value.get('cpu_ms', 51) > 50):
                issues.append('Official test incomplete or exceeds CPU limit')
            weights = ROOT/'weights/final_model.pth'
            if weights.exists() and value.get('checkpoint_sha256') != sha256(weights):
                issues.append(f'Checkpoint mismatch: {name}')
    if (ROOT/'captured/metadata.json').exists():
        from src.captured import verify_captured
        try:
            verify_captured(ROOT/'captured')
        except (OSError, ValueError, KeyError) as error:
            issues.append(str(error))
        index_file = ROOT/'results/captured_figures/index.json'
        if index_file.exists():
            records = read_json(index_file)
            photos = read_json(ROOT/'captured/metadata.json')['photos']
            if {r['id'] for r in records} != {r['id'] for r in photos} or len(records) != len(photos):
                issues.append('Captured report figures do not cover every photograph exactly once')
            for row in records:
                if not (ROOT/'results/captured_figures'/Path(row['figure']).name).is_file():
                    issues.append(f"Missing captured figure: {row['id']}")
            provenance_file = ROOT/'results/captured_figures/provenance.json'
            weights = ROOT/'weights/final_model.pth'
            if provenance_file.exists() and weights.exists():
                if read_json(provenance_file)['checkpoint_sha256'] != sha256(weights):
                    issues.append('Captured figure provenance does not match the final model')
    tests_file = ROOT/'results/tests.json'
    if tests_file.exists():
        evidence = read_json(tests_file)
        if not evidence.get('passed'):
            issues.append('Verification suite did not pass')
        for file, digest in evidence.get('source_sha256', {}).items():
            if sha256(ROOT/file) != digest:
                issues.append(f'Source changed after verification: {file}')
    return dict(missing=missing, issues=issues, automated_ready=not missing and not issues,
                manual_review_required=['phone authenticity/condition coverage', 'all report figures and metrics',
                                        'two evidence-based failure explanations', 'PDF visual inspection'])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--submission', action='store_true')
    args = p.parse_args()
    source = audit_source()
    write_json(ROOT/'results/source_audit.json', source)
    if source['issues']:
        raise SystemExit('\n'.join(source['issues']))
    print(f"Source audit passed: {len(source['imports'])} import occurrences")
    if args.submission:
        status = submission_status()
        write_json(ROOT/'results/submission_audit.json', status)
        print(status)
        if not status['automated_ready']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
