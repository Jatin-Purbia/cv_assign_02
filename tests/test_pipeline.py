from src import runtime
runtime.configure()
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tests import helpers as tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import torch
from src.dataset import paired_files, split_ids, SegmentationDataset
from src.inference import Predictor, export_weights
from src.losses import segmentation_loss, soft_dice_loss
from src.models import build_model
from src.preprocessing import binary_mask, image_tensor, read_image, read_pair
from src.trainer import train
from src.evaluation import evaluate
from src.utils import sha256, write_json
from src.inference import verify_frozen

TEST_TMP = Path(__file__).resolve().parents[1]/'tmp'/'tests'
TEST_TMP.mkdir(parents=True, exist_ok=True)


class PreprocessingTests(unittest.TestCase):
    def test_threshold(self):
        result = binary_mask(np.array([[0, 127], [128, 255]], dtype=np.uint8))
        self.assertEqual(result.shape, (256, 256))
        self.assertEqual([int(result[0, 0]), int(result[0, -1]), int(result[-1, 0]), int(result[-1, -1])], [0, 0, 1, 1])

    def test_channels_range_and_shapes(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        image[:, :, 2] = 255
        tensor = image_tensor(image)
        self.assertEqual(tuple(tensor.shape), (3, 256, 256))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(float(tensor[0].mean()), 1)
        self.assertEqual(float(tensor[2].mean()), 0)
        with self.assertRaises(ValueError):
            image_tensor(image[:100])

    def test_interpolation_contract(self):
        with tempfile.temporary_directory(dir=TEST_TMP) as temp:
            path = Path(temp)/'x.png'
            image = np.arange(15*9*3, dtype=np.uint8).reshape(15, 9, 3)
            cv2.imwrite(str(path), image)
            self.assertTrue(np.array_equal(read_image(path), cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)))

    def test_mismatched_original_dimensions(self):
        with tempfile.temporary_directory(dir=TEST_TMP) as temp:
            image, mask = Path(temp)/'x.png', Path(temp)/'mask.png'
            cv2.imwrite(str(image), np.zeros((5, 6, 3), np.uint8))
            cv2.imwrite(str(mask), np.zeros((6, 5), np.uint8))
            with self.assertRaisesRegex(ValueError, 'dimensions differ'):
                read_pair(image, mask)


class ModelLossTests(unittest.TestCase):
    def test_candidates_forward_backward(self):
        for kind in ('tiny', 'unet', 'depthwise'):
            for up in ('bilinear', 'transpose'):
                model = build_model(dict(kind=kind, base=4, depth=3, upsampling=up))
                x = torch.rand(2, 3, 33, 35)
                logits = model(x)
                self.assertEqual(tuple(logits.shape), (2, 1, 33, 35))
                loss = segmentation_loss(logits, torch.zeros_like(logits), .5)
                loss.backward()
                self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))

    def test_analytic_loss(self):
        logits, target = torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2)
        self.assertAlmostEqual(float(segmentation_loss(logits, target)), float(torch.log(torch.tensor(2.))), places=6)
        self.assertAlmostEqual(float(soft_dice_loss(logits, target)), 1/3, places=5)
        self.assertTrue(torch.isfinite(segmentation_loss(torch.full_like(logits, -1000), target)))


class DatasetTests(unittest.TestCase):
    def test_split(self):
        ids = [f'id{i}' for i in range(10553)]
        train_ids, val_ids = split_ids(ids)
        self.assertEqual((len(train_ids), len(val_ids)), (9498, 1055))
        self.assertFalse(set(train_ids) & set(val_ids))
        self.assertEqual(set(train_ids) | set(val_ids), set(ids))
        self.assertEqual((train_ids, val_ids), split_ids(list(reversed(ids))))

    def test_missing_duplicate_and_corrupt(self):
        with tempfile.temporary_directory(dir=TEST_TMP) as temp:
            images, masks = Path(temp)/'images', Path(temp)/'masks'
            images.mkdir()
            masks.mkdir()
            cv2.imwrite(str(images/'x.png'), np.zeros((4, 4, 3), np.uint8))
            with self.assertRaises(ValueError):
                paired_files(images, masks)
            cv2.imwrite(str(masks/'x.png'), np.zeros((4, 4), np.uint8))
            dataset = SegmentationDataset(images, masks)
            self.assertEqual(tuple(dataset[0][0].shape), (3, 256, 256))
            (images/'x.png').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                dataset[0]
            (images/'x.jpg').write_bytes(b'duplicate')
            with self.assertRaises(ValueError):
                paired_files(images, masks)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_TMP)
        self.root = Path(self.temp.name)
        self.weights = self.root/'test_only_random.pth'
        torch.manual_seed(42)
        config = dict(kind='tiny', base=4, depth=2)
        self.model = build_model(config).eval()
        export_weights(self.weights, self.model, config, 64)
        self.predictor = Predictor(self.weights)
        self.image = np.zeros((256, 256, 3), np.uint8)

    def tearDown(self):
        self.temp.cleanup()

    def test_roundtrip_shape_and_determinism(self):
        torch.set_num_threads(4)
        self.predictor = Predictor(self.weights)
        self.assertEqual(torch.get_num_threads(), 1)
        mask = self.predictor(self.image)
        self.assertEqual(mask.shape, (256, 256))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(set(np.unique(mask)) <= {0, 1})
        self.assertTrue(np.array_equal(mask, self.predictor(self.image)))
        for key, tensor in self.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, self.predictor.model.state_dict()[key]))
        self.assertEqual(self.predictor.size_bytes, os.path.getsize(self.weights))
        self.assertFalse(self.predictor.model.training)
        self.assertEqual(next(self.predictor.model.parameters()).device.type, 'cpu')

    def test_timing_wraps_full_call(self):
        with patch('src.inference.time.process_time', side_effect=[10., 10.0125]) as timer:
            mask, ms = self.predictor.timed(self.image)
        self.assertEqual(timer.call_count, 2)
        self.assertAlmostEqual(ms, 12.5)
        self.assertTrue(np.array_equal(mask, self.predictor(self.image)))

    def test_cli_exact_stdout_fresh_process(self):
        source, dest = self.root/'input.png', self.root/'out.png'
        cv2.imwrite(str(source), self.image)
        command = [sys.executable, 'predict.py', '--image', str(source), '--out', str(dest), '--weights', str(self.weights)]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertRegex(lines[0], r'^T_CPU_ms: \d+\.\d{2}$')
        self.assertEqual(lines[1], f'S_disk_bytes: {self.weights.stat().st_size}')
        saved = cv2.imread(str(dest), cv2.IMREAD_GRAYSCALE)
        self.assertTrue(np.array_equal(saved, self.predictor(self.image)*255))

    def test_evaluation_uses_identical_prediction(self):
        photos, masks = self.root/'photos', self.root/'masks'
        photos.mkdir()
        masks.mkdir()
        cv2.imwrite(str(photos/'x.png'), self.image)
        cv2.imwrite(str(masks/'x.png'), self.predictor(self.image)*255)
        result = evaluate(self.weights, photos, masks, self.root/'result.json', 'synthetic')
        self.assertEqual(result['miou'], 1)
        self.assertEqual(result['bf1'], 1)
        self.assertFalse(result['official'])
        with self.assertRaises(FileNotFoundError):
            evaluate(self.weights, photos, masks, self.root/'test.json', 'duts-test')

    def test_thread_environment(self):
        self.assertTrue(all(os.environ[v] == '1' for v in runtime.THREAD_VARIABLES))
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual(torch.get_num_interop_threads(), 1)
        self.assertEqual(cv2.getNumThreads(), 1)

    def test_frozen_hash_and_official_count_guards(self):
        write_json(self.weights.with_suffix('.freeze.json'), dict(sha256=sha256(self.weights)))
        self.assertEqual(verify_frozen(self.weights)['sha256'], sha256(self.weights))
        images, masks = self.root/'images', self.root/'labels'
        images.mkdir()
        masks.mkdir()
        cv2.imwrite(str(images/'x.png'), self.image)
        cv2.imwrite(str(masks/'x.png'), np.zeros((256, 256), np.uint8))
        with self.assertRaisesRegex(ValueError, '5,019'):
            evaluate(self.weights, images, masks, self.root/'final.json', 'duts-test')
        write_json(self.weights.with_suffix('.freeze.json'), dict(sha256='incorrect'))
        with self.assertRaisesRegex(ValueError, 'changed'):
            verify_frozen(self.weights)

    def test_cli_exactly_twenty_timed_runs_after_warmup(self):
        import predict
        fake = unittest.mock.Mock()
        fake.return_value = np.zeros((256, 256), np.uint8)
        fake.size_bytes = 12345
        fake.timed.return_value = (fake.return_value, 2.5)
        source, output = self.root/'source.png', self.root/'prediction.png'
        cv2.imwrite(str(source), self.image)
        with patch('predict.Predictor', return_value=fake), patch.object(sys, 'argv',
                   ['predict.py', '--image', str(source), '--out', str(output)]), contextlib.redirect_stdout(io.StringIO()) as stdout:
            predict.main()
        self.assertEqual(fake.call_count, 1)
        self.assertEqual(fake.timed.call_count, 20)
        self.assertEqual(stdout.getvalue(), 'T_CPU_ms: 2.50\nS_disk_bytes: 12345\n')


class TrainingSmokeTests(unittest.TestCase):
    def test_real_optimizer_checkpoint_loop_on_synthetic_data(self):
        with tempfile.temporary_directory(dir=TEST_TMP) as temp:
            root = Path(temp)
            tr = root/'DUTS-TR'
            images, masks = tr/'DUTS-TR-Image', tr/'DUTS-TR-Mask'
            images.mkdir(parents=True)
            masks.mkdir()
            for index in range(4):
                image, mask = np.zeros((20, 30, 3), np.uint8), np.zeros((20, 30), np.uint8)
                image[5:15, 8:22], mask[5:15, 8:22] = 220, 255
                cv2.imwrite(str(images/f'{index}.png'), image)
                cv2.imwrite(str(masks/f'{index}.png'), mask)
            (root/'train.txt').write_text('0\n1\n2\n')
            (root/'val.txt').write_text('3\n')
            cfg = dict(seed=42, data_root=str(root), train_manifest=str(root/'train.txt'),
                       val_manifest=str(root/'val.txt'), model=dict(kind='tiny', base=4, depth=2),
                       resolution=32, epochs=1, batch_size=2, lr=.001, weight_decay=.0001,
                       dice_weight=.5, gradient_clip=5, patience=1, synthetic_smoke=True)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = train(cfg, root/'run')
            self.assertEqual(summary['completed_epochs'], 1)
            self.assertTrue((root/'run'/'history.csv').exists())
            predictor = Predictor(root/'run'/'best.pth')
            self.assertEqual(predictor(np.zeros((256, 256, 3), np.uint8)).shape, (256, 256))
