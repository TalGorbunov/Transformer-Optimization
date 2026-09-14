"""CPU-Slurm checks of the V9 paired schedule and adversarial metadata cases."""
from pathlib import Path
import os
import sys
import unittest

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.native_vision_v9_pairs import self_test


class V9PairedScheduleTests(unittest.TestCase):
    def test_full_weighted_schedule_and_rejections(self):
        self.assertTrue(os.environ.get('SLURM_JOB_ID'), 'Run metadata workload in Slurm CPU')
        self.assertEqual(os.environ.get('SLURM_JOB_PARTITION'),'cpu')
        self.assertFalse(os.environ.get('SLURM_JOB_GPUS'))
        result=self_test()
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['tests']),13)


if __name__=='__main__':unittest.main()
