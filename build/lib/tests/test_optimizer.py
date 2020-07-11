import sys
import time
from unittest import TestCase, main
from random import randint
from typing import Dict, Any

sys.path.insert(0, "..")
from src import Optimizer


class DummyOptimizer(Optimizer):
    r"""A test optimizers. Generate only a dummy candidate
    to test asyncronization and networking"""

    def generate_candidate(self, trainer_info: Dict) -> Dict[str, Any]:
        time.sleep(2)
        return {"batch_size": 4}


class TestOptimizer(TestCase):
    def test_run(self):
        optimizer = DummyOptimizer()
        print(optimizer.generate_candidate({}))
        optimizer.run()
        
if __name__ == "__main__":
    main()
