from typing import Dict, Any
from argparse import Namespace

import torch

from dopt import Trainer
from dopt.synthetic_trainers import NegHartmannTrainer
from test_objective_function import run_train_net_kfold


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
            conv1=round(candidate["conv1"]),
            conv1_kernel=round(candidate["conv1_kernel"]),
            conv2=round(candidate["conv2"]),
            conv2_kernel=round(candidate["conv2_kernel"]),
            dropout1=candidate["dropout1"],
            maxpool1=round(candidate["maxpool1"]),
            maxpool2=round(candidate["maxpool2"]),
            no_cuda=False, 
            seed=1, 
            batch_size=round(candidate["batch_size"]),
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
    trainer = NegHartmannTrainer(host="jvs008-r1.bucknell.edu",
                                 port="15555")
    trainer.run()
    