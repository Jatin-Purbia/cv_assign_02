"""Qualification must reject regressions and fail closed on invalid manifests."""
from src import runtime
runtime.configure()
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
from tests.helpers import temporary_directory
from src.deployment import active_weights, release_gates
from src.inference import export_weights
from src.models import build_model
from src.utils import write_json, sha256, environment


class DeploymentTests(unittest.TestCase):
    def results(self):
        value = dict(split='validation', manifest_sha256='same', images=4, q=.7, miou=.8,
                     bf1=.5, cpu_ms=20, per_image=[dict(id=str(i), q=.7, bf1=.5, cpu_ms=20,
                     confusion_matrix=[[75, 5], [5, 15]]) for i in range(4)])
        candidate = copy.deepcopy(value)
        candidate.update(q=.72, bf1=.55)
        for record in candidate['per_image']:
            record.update(bf1=.55, confusion_matrix=[[77, 3], [3, 17]])
        return value, candidate

    def test_quality_latency_and_regression_gates(self):
        before, after = self.results()
        self.assertTrue(release_gates(before, after)['passed'])
        for field, value in [('q', .701), ('miou', .79), ('bf1', .49), ('cpu_ms', 41)]:
            bad = dict(after, **{field: value})
            self.assertFalse(release_gates(before, bad)['passed'], field)
        bad = copy.deepcopy(after)
        bad['per_image'][0]['cpu_ms'] = 51
        self.assertFalse(release_gates(before, bad)['passed'])
        bad = copy.deepcopy(after)
        bad['per_image'][0]['q'] = bad['per_image'][1]['q'] = .5
        self.assertFalse(release_gates(before, bad)['passed'])

    def test_mismatched_or_test_evidence_rejected(self):
        before, after = self.results()
        for field, value in [('manifest_sha256', 'changed'), ('split', 'duts-test'), ('images', 5)]:
            with self.assertRaises(ValueError):
                release_gates(before, dict(after, **{field: value}))

    def test_active_pointer_requires_valid_gate_and_checkpoint_hash(self):
        with temporary_directory(Path('tmp/tests')) as folder:
            root = Path(folder)
            with patch('src.deployment.ROOT', root):
                self.assertEqual(active_weights(), root/'weights/final_model.pth')
                weights = root/'weights/releases/test/model.pth'
                cfg = dict(kind='tiny', base=4, depth=2)
                export_weights(weights, build_model(cfg), cfg, 32)
                write_json(weights.with_suffix('.freeze.json'), dict(sha256=sha256(weights)))
                pointer = dict(format_version=1, task='automatic-salient-object',
                               weights='weights/releases/test/model.pth', sha256=sha256(weights),
                               gates=dict(passed=True), stress_gates=dict(passed=True))
                write_json(root/'weights/active_model.json', pointer)
                self.assertEqual(active_weights(), weights.resolve())
                pointer['validated_environment'] = dict(environment(), torch='unsupported-fixture-runtime')
                write_json(root/'weights/active_model.json', pointer)
                with self.assertRaisesRegex(ValueError, 'validated runtime'):
                    active_weights()
                pointer.pop('validated_environment')
                pointer['gates']['passed'] = False
                write_json(root/'weights/active_model.json', pointer)
                with self.assertRaisesRegex(ValueError, 'qualification'):
                    active_weights()
                pointer['gates']['passed'] = True
                pointer['weights'] = '../outside.pth'
                write_json(root/'weights/active_model.json', pointer)
                with self.assertRaisesRegex(ValueError, 'under weights/releases'):
                    active_weights()
                pointer['weights'] = 'weights/releases/test/model.pth'
                write_json(root/'weights/active_model.json', pointer)
                weights.write_bytes(weights.read_bytes()+b'changed')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    active_weights()

    def test_prediction_rejects_overwriting_input(self):
        import predict
        with patch('sys.argv', ['predict.py', '--image', 'same.png', '--out', 'same.png']):
            with self.assertRaisesRegex(ValueError, 'must differ'):
                predict.main()

    def test_folder_prediction_preserves_masks_when_photo_is_named_comparison(self):
        import cv2
        import numpy as np
        from scripts.predict_folder import predict_folder
        with temporary_directory(Path('tmp/tests')) as folder:
            root = Path(folder)
            images = root/'images'
            images.mkdir()
            cv2.imwrite(str(images/'comparison.png'), np.zeros((256, 256, 3), np.uint8))
            cfg = dict(kind='tiny', base=4, depth=2)
            weights = root/'model.pth'
            export_weights(weights, build_model(cfg), cfg, 32)
            records = predict_folder(images, root/'output', weights)
            mask = cv2.imread(str(root/'output'/records[0]['mask']), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(mask.shape, (256, 256))
            self.assertTrue(set(np.unique(mask)) <= {0, 255})
            self.assertTrue((root/'output/comparison.png').is_file())
            with self.assertRaises(FileExistsError):
                predict_folder(images, root/'output', weights)

    def test_qualification_activates_only_after_passing_evidence_is_saved(self):
        from scripts.qualify_model import qualify
        from src.utils import read_json
        with temporary_directory(Path('tmp/tests')) as folder:
            root = Path(folder).resolve()
            cfg = dict(kind='tiny', base=4, depth=2)
            model = build_model(cfg)
            export_weights(root/'weights/final_model.pth', model, cfg, 96)
            manifest = root/'data/splits/val.txt'
            manifest.parent.mkdir(parents=True)
            manifest.write_text('a\nb\nc\nd\n')
            for name in ('inference', 'preprocessing', 'models', 'runtime', 'deployment', 'robustness', 'statistics', 'evaluation', 'optimized_inference'):
                file = root/'src'/f'{name}.py'
                file.parent.mkdir(exist_ok=True)
                file.write_text('# frozen fixture source\n')
            (root/'scripts').mkdir()
            (root/'scripts/qualify_model.py').write_text('# qualification fixture\n')
            run = root/'experiments/fine'
            export_weights(run/'best.pth', model, cfg, 128)
            write_json(run/'run.json', dict(config=dict(synthetic_smoke=False),
                       val_manifest_sha256=sha256(manifest)))
            write_json(run/'summary.json', dict(finished=True))

            def evaluation(weights, images, masks, output, split, ids):
                before, after = self.results()
                result = before if Path(weights).name == 'final_model.pth' else after
                result.update(manifest_sha256=sha256(manifest), checkpoint_sha256=sha256(weights),
                              size_bytes=Path(weights).stat().st_size, ces=.65, environment=environment())
                write_json(output, result)
                return result

            def stress(weights):
                q = .7 if Path(weights).name == 'final_model.pth' else .72
                return dict(ids=['a'], manifest_sha256=sha256(manifest), mean_q=q,
                            conditions={k: dict(q=q) for k in ('dark', 'bright', 'low_contrast')})

            import contextlib
            import io
            with patch('scripts.qualify_model.ROOT', root), patch('src.deployment.ROOT', root), \
                 patch('src.inference.ROOT', root), patch('scripts.qualify_model.evaluate', side_effect=evaluation), \
                 patch('scripts.qualify_model.stress_validation', side_effect=stress), \
                 contextlib.redirect_stdout(io.StringIO()):
                report = qualify([str(run)], root/'results/quality', activate=True)
                self.assertTrue(report['activated'])
                selected = active_weights()
                self.assertTrue(selected.is_file())
                pointer = read_json(root/'weights/active_model.json')
                self.assertTrue((root/pointer['qualification']).is_file())
                self.assertEqual(sha256(selected), pointer['sha256'])
                saved_pointer = (root/'weights/active_model.json').read_bytes()
                def failing_stress(weights):
                    result = stress(weights)
                    if Path(weights).name != 'final_model.pth':
                        result['mean_q'] = .5
                        result['conditions']['dark']['q'] = .5
                    return result
                with patch('scripts.qualify_model.stress_validation', side_effect=failing_stress):
                    rejected = qualify([str(run)], root/'results/rejected', activate=True)
                self.assertFalse(rejected['activated'])
                self.assertIsNone(rejected['selected'])
                self.assertEqual((root/'weights/active_model.json').read_bytes(), saved_pointer)
