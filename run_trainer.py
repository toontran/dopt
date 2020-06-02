from typing import Dict, Any
from time import sleep

import torch
from botorch.test_functions.synthetic import Hartmann

from src.synthetic_trainers.neghartmann_trainer import NegHartmannTrainer


if __name__ == "__main__":
    trainer = NegHartmannTrainer(host="jvs008-r1.bucknell.edu",
                                 port="15555")
    trainer.run()
    