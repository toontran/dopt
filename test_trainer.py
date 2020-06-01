from typing import Dict, Any
from time import sleep

import torch
from botorch.test_functions.synthetic import Hartmann

from src.trainer import Trainer


class NegHartmannTrainer(Trainer):
    
    NOISE_SE = 0.5
    NOISE_OF_NOISE = 0.1
    NOISE_FURTHER = 0.2
    
    def get_observation(self, candidate: Dict[str, Any]) \
            -> Dict[str, Any]:
        r""" Get observation by plugging the candidate into objective function.
        This method is made abstract to easier modify the objective function
        to run on different platforms.

        :param candidate:
        :return:
        """
        print(f"Trainer received: {candidate}")
        objective_function = Hartmann(negate=True)
        train_x = torch.tensor([candidate["x1"], candidate["x2"], candidate["x3"],
                                candidate["x4"], candidate["x5"], candidate["x6"]])
        exact_objective = objective_function(train_x).unsqueeze(-1)  # add output dimension
        # Add noise to the objective function
        observed_noise = NegHartmannTrainer.NOISE_SE * torch.randn_like(exact_objective) + \
                         NegHartmannTrainer.NOISE_OF_NOISE * torch.randn_like(exact_objective)
        observed_variance = observed_noise ** 2
        noisy_objective = exact_objective + observed_noise + \
                          NegHartmannTrainer.NOISE_FURTHER * torch.randn_like(exact_objective)
        return (
            float(noisy_objective.cpu().numpy()[0]), 
            float(observed_variance.cpu().numpy()[0])
        )
    

class ToyFunctionTrainer(Trainer):
    
    NOISE_SE = 0.5
    NOISE_OF_NOISE = 0.1
    NOISE_FURTHER = 0.2
    
    def get_observation(self, candidate: Dict[str, Any]) \
            -> Dict[str, Any]:
        r""" Get observation by plugging the candidate into objective function.
        This method is made abstract to easier modify the objective function
        to run on different platforms.

        :param candidate:
        :return:
        """
        train_x = torch.tensor([candidate["x1"], candidate["x2"]], dtype=float)
        exact_objective = train_x[0] ** 2 - (train_x[1] - 1) ** 2 + 1
        observed_noise = ToyFunctionTrainer.NOISE_SE * torch.randn_like(exact_objective) + \
                         ToyFunctionTrainer.NOISE_OF_NOISE * torch.randn_like(exact_objective)
        noisy_objective = exact_objective + observed_noise + \
                          ToyFunctionTrainer.NOISE_FURTHER * torch.randn_like(exact_objective)
        sleep(60)
        return (
            float(noisy_objective.cpu().numpy()), 
            float(observed_noise.cpu().numpy())
        )
  

if __name__ == "__main__":
    trainer = NegHartmannTrainer(host="jvs008-r1.bucknell.edu",
                                 port="15555")
    trainer.run()
    