from unittest import TestCase
from src import Optimizer
from random import randint


class ConstantOptimizer(Optimizer):
    r"""A test optimizer. Generate only a dummy candidate
    to test asyncronization and networking"""

    def generate_candidate(self):
        return randint(1, 4)


class TestOptimizer(TestCase):
    def test_run(self):
        optim = ConstantOptimizer(max_observations=3)
        optim.run()
