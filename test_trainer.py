from typing import Dict, Any

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
        noisy_objective = exact_objective + observed_noise + \
                          NegHartmannTrainer.NOISE_FURTHER * torch.randn_like(exact_objective)
        return {"result": (
            float(noisy_objective.cpu().numpy()[0]), 
            float(observed_noise.cpu().numpy()[0])
        )}
  

if __name__ == "__main__":
    trainer = NegHartmannTrainer()
    trainer.run()
    