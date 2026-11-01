from src import runtime
runtime.configure()
import copy
import unittest
import numpy as np
from src.robustness import photometric, stress_gates


class RobustnessTests(unittest.TestCase):
    def test_photometric_transforms_are_bounded_and_keep_input(self):
        image = np.array([[[0, 100, 255], [255, 100, 0]]], dtype=np.uint8)
        saved = image.copy()
        self.assertEqual(photometric(image, 'dark')[0, 0].tolist(), [0, 60, 153])
        self.assertEqual(photometric(image, 'bright')[0, 0].tolist(), [0, 130, 255])
        contrast = photometric(image, 'low_contrast')
        self.assertGreater(int(contrast.min()), 0)
        self.assertLess(int(contrast.max()), 255)
        self.assertTrue(np.array_equal(image, saved))

    def test_stress_gate_rejects_hidden_condition_regression(self):
        before = dict(ids=['a'], manifest_sha256='same', mean_q=.7,
                      conditions={k: dict(q=.7) for k in ('dark', 'bright', 'low_contrast')})
        after = copy.deepcopy(before)
        after['mean_q'] = .72
        for value in after['conditions'].values():
            value['q'] = .72
        self.assertTrue(stress_gates(before, after)['passed'])
        after['conditions']['dark']['q'] = .68
        self.assertFalse(stress_gates(before, after)['passed'])
