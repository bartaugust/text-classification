import pytest
import warnings
import logging

import hydra
from hydra.test_utils.test_utils import TSweepRunner

logger = logging.getLogger(__name__)


@pytest.fixture
def sweep(hydra_sweep_runner: TSweepRunner):
    sweeper = hydra_sweep_runner(calling_file=None,
                                 calling_module='hydra.test_utils.a_module',
                                 config_path='../../conf',
                                 config_name='config',
                                 task_function=None,
                                 overrides=[])
    return sweeper

def test_with_sweep(sweep):
    with sweep:
        job_ret = sweep.returns[0]
        for job in job_ret:
            cfg = job.cfg
