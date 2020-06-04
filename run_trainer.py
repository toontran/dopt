from typing import Dict, Any
from time import sleep
from argparse import Namespace

import torch
from botorch.test_functions.synthetic import Hartmann

from src.trainer import Trainer
# from src.synthetic_trainers.neghartmann_trainer import NegHartmannTrainer
from test_yaleface_objective import run_train_net_kfold


class YaleFaceTrainer(Trainer):
    def get_observation(self, candidate: Dict[str, Any]) \
            -> Dict[str, Any]:
        r""" Get observation by plugging the candidate into objective function.
        This method is made abstract to easier modify the objective function
        to run on different platforms.

        :param candidate:
        :return:
        """
        args = Namespace(
            no_cuda=False, 
            seed=1, 
            batch_size=candidate["batch_size"],
            test_batch_size=1000,
            epochs=23,
            lr=candidate["lr"],
            gamma=0.7,
            log_interval=250, # was 250
            save_model=False,
            num_folds = 5
        )
        
        mean, variance = run_train_net_kfold(args)
        return mean, variance


if __name__ == "__main__":
    trainer = YaleFaceTrainer(host="jvs008-r1.bucknell.edu",
                                 port="15555")
    trainer.run()
    