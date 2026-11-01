"""Deferred qualification must wait for training, or leave the release alone."""
import contextlib
import io
import unittest
from unittest.mock import patch
from scripts import qualify_model


class QualificationCliTests(unittest.TestCase):
    def test_resolution_only_qualification(self):
        with patch('sys.argv', ['qualify_model', '--out', 'results/check']), \
             patch.object(qualify_model, 'qualify', return_value={'selected': None}) as qualify, \
             contextlib.redirect_stdout(io.StringIO()):
            qualify_model.main()
        self.assertEqual(qualify.call_args.args[0], [])

    def test_waits_for_completed_summary_before_qualification(self):
        with patch('sys.argv', ['qualify_model', '--runs', 'run', '--wait-hours', '1']), \
             patch.object(qualify_model.Path, 'exists', side_effect=[False, True]), \
             patch.object(qualify_model, 'read_json', return_value={'finished': True}), \
             patch.object(qualify_model.time, 'sleep') as sleep, \
             patch.object(qualify_model, 'qualify', return_value={'selected': None}) as qualify, \
             contextlib.redirect_stdout(io.StringIO()):
            qualify_model.main()
        sleep.assert_called_once_with(10)
        qualify.assert_called_once()

    def test_timeout_does_not_qualify_or_activate(self):
        with patch('sys.argv', ['qualify_model', '--runs', 'run', '--wait-hours', '1', '--activate']), \
             patch.object(qualify_model.Path, 'exists', return_value=False), \
             patch.object(qualify_model.time, 'monotonic', side_effect=[0, 3601]), \
             patch.object(qualify_model, 'qualify') as qualify, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(TimeoutError):
                qualify_model.main()
        qualify.assert_not_called()
