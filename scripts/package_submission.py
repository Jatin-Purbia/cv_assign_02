"""Copy an allowlisted clean folder after the submission audit passes. No compression."""
from src import runtime
runtime.configure()
import argparse
from pathlib import Path
import shutil
from scripts.audit import audit_source, submission_status
from src.utils import ROOT, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='submission/B23CH1022_Assignment2')
    p.add_argument('--draft', action='store_true', help='Allow a clearly marked incomplete review snapshot')
    args = p.parse_args()
    status = submission_status()
    if audit_source()['issues'] or (not status['automated_ready'] and not args.draft):
        raise ValueError(f'Submission incomplete: {status}')
    target = Path(args.out).resolve()
    if not target.is_relative_to((ROOT/'submission').resolve()) or target.exists():
        raise ValueError('Choose a new output folder under the project submission/ directory')
    target.mkdir(parents=True)
    for name in ('src', 'scripts', 'tests', 'configs', 'docs', 'captured', 'weights', 'report', 'results'):
        shutil.copytree(ROOT/name, target/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.aux', '*.log', '*.out'))
    for name in ('README.md', 'requirements.txt', 'predict.py', 'train.py', 'evaluate.py', 'benchmark.py'):
        shutil.copyfile(ROOT/name, target/name)
    shutil.copytree(ROOT/'data/splits', target/'data/splits')
    shutil.copyfile(ROOT/'data/README.md', target/'data/README.md')
    if (ROOT/'CV_Assignment2.pdf').exists():
        shutil.copyfile(ROOT/'CV_Assignment2.pdf', target/'CV_Assignment2.pdf')
    (target/'experiments').mkdir(exist_ok=True)
    shutil.copyfile(ROOT/'experiments/README.md', target/'experiments/README.md')
    for run in (ROOT/'experiments').rglob('run.json'):
        if not (run.parent/'summary.json').exists():
            continue
        for record in run.parent.iterdir():
            if record.is_file() and record.suffix in {'.json', '.csv'}:
                dest = target/record.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(record, dest)
    write_json(target/'submission_audit.json', status)
    if args.draft:
        (target/'DRAFT_INCOMPLETE.md').write_text(
            '# INCOMPLETE REVIEW SNAPSHOT\n\nNot a final submission. Consult submission_audit.json and docs/STATUS.md for mandatory missing work.\n',
            encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
