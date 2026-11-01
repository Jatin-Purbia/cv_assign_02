from src import runtime
runtime.configure()
import copy
import unittest
from src.statistics import paired_quality_interval


class StatisticsTests(unittest.TestCase):
    def test_positive_negative_and_identical_paired_intervals(self):
        before = dict(per_image=[dict(id=str(i), bf1=.5,
                      confusion_matrix=[[75, 5], [5, 15]]) for i in range(5)])
        after = copy.deepcopy(before)
        for record in after['per_image']:
            record.update(bf1=.7, confusion_matrix=[[78, 2], [2, 18]])
        positive = paired_quality_interval(before, after, draws=100)
        self.assertTrue(positive['passed'])
        self.assertGreater(positive['q_gain_95_interval'][0], 0)
        self.assertFalse(paired_quality_interval(after, before, draws=100)['passed'])
        self.assertEqual(paired_quality_interval(before, before, draws=100)['q_gain_95_interval'], [0., 0.])

    def test_changed_ground_truth_rejected(self):
        before = dict(per_image=[dict(id='x', bf1=.5, confusion_matrix=[[75, 5], [5, 15]])])
        after = dict(per_image=[dict(id='x', bf1=.7, confusion_matrix=[[78, 1], [2, 19]])])
        with self.assertRaisesRegex(ValueError, 'target class counts'):
            paired_quality_interval(before, after, draws=100)
