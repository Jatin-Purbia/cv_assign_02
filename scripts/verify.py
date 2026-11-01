"""Run the complete suite and save honest machine-readable verification evidence."""
from src import runtime
runtime.configure()
import io
import unittest
from datetime import datetime, timezone
from src.utils import ROOT, environment, write_json, sha256
from scripts.audit import audit_source


def main() -> None:
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), top_level_dir=str(ROOT))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/test_output.txt').write_text(stream.getvalue(), encoding='utf-8')
    audit = audit_source()
    write_json(ROOT/'results/source_audit.json', audit)
    write_json(ROOT/'results/tests.json', dict(tests_run=result.testsRun, passed=result.wasSuccessful(),
               failures=[str(test) for test, _ in result.failures], errors=[str(test) for test, _ in result.errors],
               tested_utc=datetime.now(timezone.utc).isoformat(), environment=environment(),
               evidence='Analytic arrays and synthetic fixtures; not DUTS performance',
               source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                   list(ROOT.glob('*.py')) + list((ROOT/'src').rglob('*.py'))
                   + list((ROOT/'scripts').rglob('*.py')) + list((ROOT/'tests').rglob('*.py'))}))
    print(f'Tests: {result.testsRun}; failures: {len(result.failures)}; errors: {len(result.errors)}')
    print(f"Source audit issues: {len(audit['issues'])}")
    if not result.wasSuccessful() or audit['issues']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
