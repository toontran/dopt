from unittest import TestCase
from random import randint
from typing import Dict

from src import Optimizer


class ConstantOptimizer(Optimizer):
    r"""A test optimizer. Generate only a dummy candidate
    to test asyncronization and networking"""

    def generate_candidate(self, trainer_info: Dict):
        return randint(1, 4)


class TestOptimizer(TestCase):
    def test_run(self):
        optimizer = ConstantOptimizer(max_observations=3)
        print(optimizer.generate_candidate({}))
        optimizer.run()
