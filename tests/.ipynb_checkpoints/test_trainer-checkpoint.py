import sys
import time
from abc import ABC
from typing import Dict, Any
from unittest import TestCase, main

sys.path.insert(0, "..")
from src import Trainer


class DummyTrainer(Trainer):

    def get_observation(self, candidate: Dict[str, Any])\
            -> Dict[str, Any]:
        time.sleep(5)
        return {"result": 3}


class TestTrainer(TestCase):
    def test_run(self):
        trainer = DummyTrainer()
        print(trainer.get_observation({"batch_size": 0}))
        trainer.run()

        
if __name__ == "__main__":
    main()