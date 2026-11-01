from src import runtime
runtime.configure()
import unittest
import torch
from src.metrics import boundary, boundary_f1, MetricAccumulator, quality, ces, _nearby


class MetricsTests(unittest.TestCase):
    def test_vector_window_matches_pooling_and_analytic_corner(self):
        corner = torch.zeros(7, 7, dtype=torch.bool)
        corner[0, 0] = True
        expected = torch.zeros_like(corner)
        expected[:3, :3] = True
        self.assertTrue(torch.equal(_nearby(corner), expected))
        generator = torch.Generator().manual_seed(73)
        for shape in [(1, 1), (2, 7), (16, 19), (256, 256)]:
            for density in (0., .01, .1, 1.):
                b = torch.rand(shape, generator=generator) < density
                reference = torch.nn.functional.max_pool2d(b.float()[None, None], 5, 1, 2)[0, 0].bool()
                self.assertTrue(torch.equal(_nearby(b), reference))

    def score(self, p, t):
        m = MetricAccumulator()
        m.update(p, t)
        return m.compute()

    def test_perfect_foreground_and_background(self):
        for value in (0, 1):
            with self.subTest(value=value):
                mask = torch.full((7, 9), value)
                r = self.score(mask, mask)
                self.assertEqual((r['miou'], r['bf1'], r['q']), (1, 1, 1))

    def test_full_mask_has_border_boundary(self):
        mask = torch.ones(5, 5)
        expected = torch.ones(5, 5, dtype=torch.bool)
        expected[1:-1, 1:-1] = False
        self.assertTrue(torch.equal(boundary(mask), expected))

    def test_no_boundary(self):
        self.assertEqual(int(boundary(torch.zeros(5, 5)).sum()), 0)

    def test_one_boundary_empty(self):
        empty, full = torch.zeros(5, 5), torch.ones(5, 5)
        self.assertEqual(boundary_f1(empty, full), 0)
        self.assertEqual(boundary_f1(full, empty), 0)

    def test_both_boundaries_empty(self):
        self.assertEqual(boundary_f1(torch.zeros(4, 4), torch.zeros(4, 4)), 1)

    def test_single_pixel(self):
        mask = torch.zeros(7, 7)
        mask[3, 3] = 1
        self.assertTrue(torch.equal(boundary(mask), mask.bool()))
        self.assertEqual(boundary_f1(mask, mask), 1)

    def test_border_touching(self):
        mask = torch.zeros(5, 5)
        mask[:3, :3] = 1
        expected = mask.bool()
        expected[1, 1] = False
        self.assertTrue(torch.equal(boundary(mask), expected))

    def test_window_offsets_and_corners(self):
        for dx, dy, expected in [(1, 0, 1), (2, 0, 1), (2, 2, 1), (0, 3, 0), (3, 3, 0)]:
            with self.subTest(dx=dx, dy=dy):
                p, t = torch.zeros(8, 8), torch.zeros(8, 8)
                t[0, 0], p[dy, dx] = 1, 1
                self.assertEqual(boundary_f1(p, t), expected)

    def test_nontrivial_precision_recall(self):
        p, t = torch.zeros(12, 12), torch.zeros(12, 12)
        p[0, 0] = p[10, 10] = t[2, 2] = 1
        self.assertAlmostEqual(boundary_f1(p, t), 2/3)

    def test_wrong_foreground_and_background(self):
        for value in (0, 1):
            r = self.score(torch.full((5, 5), value), torch.full((5, 5), 1-value))
            self.assertEqual(r['miou'], 0)
            self.assertEqual(r['bf1'], 0)

    def test_global_aggregation_not_per_image_average(self):
        metric = MetricAccumulator()
        # Image 1: TP=1,TN=3. Image 2: FN=4.
        metric.update(torch.tensor([[1, 0], [0, 0]]), torch.tensor([[1, 0], [0, 0]]))
        metric.update(torch.zeros(2, 2), torch.ones(2, 2))
        result = metric.compute()
        self.assertEqual(result['confusion_matrix'], [[3, 0], [4, 1]])
        self.assertAlmostEqual(result['miou'], (1/5+3/7)/2)
        self.assertEqual(result['bf1'], .5)

    def test_range_determinism_differing_objects(self):
        p, t = torch.zeros(20, 20), torch.zeros(20, 20)
        p[5:9, 5:9], t[3:12, 3:12] = 1, 1
        first = self.score(p, t)
        self.assertEqual(first, self.score(p, t))
        for key in ('iou_fg', 'iou_bg', 'miou', 'bf1', 'q'):
            self.assertTrue(0 <= first[key] <= 1)

    def test_invalid_masks(self):
        for bad in (torch.zeros(2, 3, 4), torch.full((3, 3), 255), torch.zeros(0, 3)):
            with self.assertRaises(ValueError):
                boundary(bad)
        with self.assertRaises(ValueError):
            boundary_f1(torch.zeros(2, 2), torch.zeros(3, 3))
        with self.assertRaises(ValueError):
            MetricAccumulator().compute()


class ScoringTests(unittest.TestCase):
    def test_q(self):
        self.assertAlmostEqual(quality(.8, .6), .74)

    def test_assignment_example(self):
        self.assertAlmostEqual(ces(.7, 20, 524288), .63525)

    def test_limit_and_cap(self):
        self.assertEqual(ces(1, 50.00001, 0), 0)
        self.assertAlmostEqual(ces(1, 50, 0), .8)
        self.assertEqual(ces(1, 0, 0), 1)
        self.assertEqual(ces(1, 0, 4194304), ces(1, 0, 9999999))
        self.assertAlmostEqual(ces(1, 0, 4194304), .9)

    def test_invalid_scores(self):
        for args in [(1, -1, 0), (1, 0, -1), (2, 0, 0), (float('nan'), 1, 1)]:
            with self.assertRaises(ValueError):
                ces(*args)
