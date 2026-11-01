from src import runtime
runtime.configure()
from pathlib import Path
import unittest
import cv2
import numpy as np
import torch
from tests.helpers import temporary_directory
from src.evaluation import evaluate
from src.inference import Predictor
from src.models import build_model
from src.optimized_inference import ConfiguredPredictor, export_configured_weights


class OptimizedInferenceTests(unittest.TestCase):
    def test_channels_last_preserves_network_outputs_and_state(self):
        with temporary_directory(Path('tmp/tests')) as folder:
            for kind in ('tiny', 'unet', 'depthwise'):
                torch.manual_seed(73)
                cfg = dict(kind=kind, base=4, depth=3)
                model = build_model(cfg).eval()
                weights = Path(folder)/f'{kind}.pth'
                export_configured_weights(weights, model, cfg, 32, .5, 'channels_last')
                predictor = ConfiguredPredictor(weights)
                x = torch.rand(2, 3, 33, 35)
                with torch.inference_mode():
                    torch.testing.assert_close(model(x), predictor.model(x), atol=1e-5, rtol=1e-5)
                first = next(predictor.model.parameters())
                self.assertTrue(first.is_contiguous(memory_format=torch.channels_last))
                image = np.zeros((256, 256, 3), np.uint8)
                self.assertTrue(np.array_equal(predictor(image), Predictor(weights)(image)))
                self.assertEqual(predictor.memory_format, 'channels_last')

    def test_evaluation_uses_recorded_layout_and_per_image_confusion(self):
        with temporary_directory(Path('tmp/tests')) as folder:
            root = Path(folder)
            cfg = dict(kind='tiny', base=4, depth=2)
            weights = root/'model.pth'
            export_configured_weights(weights, build_model(cfg), cfg, 32, .5, 'channels_last')
            predictor = ConfiguredPredictor(weights)
            photo = np.zeros((256, 256, 3), np.uint8)
            for name in ('images', 'masks'):
                (root/name).mkdir()
            cv2.imwrite(str(root/'images/x.png'), photo)
            cv2.imwrite(str(root/'masks/x.png'), predictor(photo)*255)
            result = evaluate(weights, root/'images', root/'masks', root/'result.json', 'synthetic')
            self.assertEqual(result['memory_format'], 'channels_last')
            self.assertEqual(result['q'], 1)
            self.assertEqual(result['per_image'][0]['confusion_matrix'], result['confusion_matrix'])

    def test_unknown_layout_is_rejected(self):
        with temporary_directory(Path('tmp/tests')) as folder:
            cfg = dict(kind='tiny', base=4, depth=2)
            weights = Path(folder)/'bad.pth'
            export_configured_weights(weights, build_model(cfg), cfg, 32, .5, 'contiguous')
            bundle = torch.load(weights, weights_only=True)
            bundle['memory_format'] = 'unsupported'
            torch.save(bundle, weights)
            with self.assertRaisesRegex(ValueError, 'memory format'):
                ConfiguredPredictor(weights)
