from src import runtime
runtime.configure()
import ast
from pathlib import Path
import unittest
from scripts.audit import audit_source
from scripts.annotate import polygon_mask


class ComplianceTests(unittest.TestCase):
    def test_source_audit(self):
        self.assertEqual(audit_source()['issues'], [])

    def test_runtime_precedes_numeric_imports(self):
        entries = [Path(n) for n in ('train.py', 'predict.py', 'evaluate.py')]
        entries += list(Path('scripts').glob('*.py'))
        for path in entries:
            tree = ast.parse(path.read_text())
            runtime_line = None
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module == 'src' and any(n.name == 'runtime' for n in node.names):
                    runtime_line = node.lineno
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('src.'):
                    self.assertIsNotNone(runtime_line, str(path))
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name.split('.')[0] in ('torch', 'numpy', 'cv2', 'matplotlib'):
                            self.assertIsNotNone(runtime_line, str(path))
                            self.assertLess(runtime_line, node.lineno)

    def test_manual_polygon_rasterization(self):
        mask = polygon_mask([(0, 0), (2, 0), (2, 2), (0, 2)])
        self.assertEqual(int(mask.sum()), 4)
        self.assertTrue(bool(mask[:2, :2].all()))
