"""Recovery must produce the same learned state as uninterrupted training."""
from src import runtime
runtime.configure()
import contextlib
import io
from pathlib import Path
import unittest
import cv2
import numpy as np
import torch
from tests.helpers import temporary_directory
from src.dataset import SegmentationDataset
from src.trainer import train
from src.inference import Predictor, export_weights
from src.models import build_model


class TrainingReliabilityTests(unittest.TestCase):
    def fixture(self, root):
        images = root/'DUTS-TR'/'DUTS-TR-Image'
        masks = root/'DUTS-TR'/'DUTS-TR-Mask'
        images.mkdir(parents=True)
        masks.mkdir()
        for index in range(4):
            y = np.zeros((256, 256), np.uint8)
            y[30+index:150, 40:180] = 255
            cv2.imwrite(str(masks/f'{index}.png'), y)
            cv2.imwrite(str(images/f'{index}.png'), np.repeat(y[:, :, None], 3, axis=2))
        (root/'train.txt').write_text('0\n1\n2\n')
        (root/'val.txt').write_text('3\n')
        return dict(seed=42, data_root=str(root), train_manifest=str(root/'train.txt'),
                    val_manifest=str(root/'val.txt'), model=dict(kind='tiny', base=4, depth=2),
                    resolution=32, epochs=2, batch_size=2, lr=.001, weight_decay=.0001,
                    dice_weight=.3, gradient_clip=5, patience=3, synthetic_smoke=True,
                    augmentation=dict(geometry_probability=.8, scale=.15, translation=.1,
                                      brightness=.2, contrast=.2, saturation=.15))

    def test_resume_matches_uninterrupted_training_and_repairs_history(self):
        with temporary_directory(Path('tmp/tests')) as folder:
            root = Path(folder)
            cfg = self.fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                train(cfg, root/'full')
                partial = train(cfg, root/'resumed', max_epochs=1)
                self.assertFalse(partial['finished'])
                self.assertFalse((root/'resumed/summary.json').exists())
                with (root/'resumed/history.csv').open('a') as handle:
                    handle.write('interrupted,uncommitted,row\n')
                complete = train(cfg, root/'resumed', resume=True)
            self.assertTrue(complete['finished'])
            a = torch.load(root/'full/last_checkpoint.pth', weights_only=True)
            b = torch.load(root/'resumed/last_checkpoint.pth', weights_only=True)
            for key in a['model']:
                self.assertTrue(torch.equal(a['model'][key], b['model'][key]), key)
            for first, second in zip(a['history'], b['history']):
                self.assertEqual({k: v for k, v in first.items() if k != 'seconds'},
                                 {k: v for k, v in second.items() if k != 'seconds'})
            self.assertNotIn('uncommitted', (root/'resumed/history.csv').read_text())
            with self.assertRaisesRegex(ValueError, 'config'):
                train(dict(cfg, lr=.1), root/'resumed', resume=True)

    def test_geometry_keeps_image_and_mask_aligned(self):
        with temporary_directory(Path('tmp/tests')) as folder:
            root = Path(folder)
            cfg = self.fixture(root)
            data = SegmentationDataset(root/'DUTS-TR/DUTS-TR-Image', root/'DUTS-TR/DUTS-TR-Mask',
                                       augment=True, augmentation=dict(geometry_probability=.999,
                                       scale=.2, translation=.2, brightness=0, contrast=0))
            for seed in range(5):
                torch.manual_seed(seed)
                x, y = data[0]
                self.assertGreater(float(((x[:1] >= .5) == y.bool()).float().mean()), .999)
                self.assertTrue(bool(((y == 0) | (y == 1)).all()))

    def test_initialization_has_exact_provenance(self):
        with temporary_directory(Path('tmp/tests')) as folder:
            root = Path(folder)
            cfg = self.fixture(root)
            initial = root/'initial.pth'
            export_weights(initial, build_model(cfg['model']), cfg['model'], 32)
            cfg.update(initialize_from=str(initial), epochs=1)
            with contextlib.redirect_stdout(io.StringIO()):
                train(cfg, root/'fine')
            from src.utils import read_json, sha256
            self.assertEqual(read_json(root/'fine/run.json')['initialization']['sha256'], sha256(initial))
            Predictor(root/'fine/best.pth')
