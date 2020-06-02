import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional, Tuple
import json
import random

import torch
import numpy as np
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.utils.errors import NanError
from botorch.models import HeteroskedasticSingleTaskGP
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim import optimize_acqf

from src.optimizers.optimizer import Optimizer


class NEIOptimizer(Optimizer):
    r"""A Bayesian Optimizer that uses Noisy Expected Improvement
    as the acquisition function.
    
    Example:
        >>> bounds = TODO
        >>> optimizer = NEIOptimizer(bounds, device="cuda:0")
        >>> optimizer.run()
    """
    MC_SAMPLES = 500
    DTYPE = torch.double
    
    def __init__(self, 
                 file_name: str,
                 bounds: Dict[str, Tuple[float, float]],
                 device: Optional[str] = "cpu",
                 seed: Optional[int] = random.randint(1, 100000)):
        r"""Constructor for  Bayesian optimizer that use Noisy Expected Improvement
        as the acquisition function. 
        
        :param device: Generate candidates on the chosen device.
        :param bounds: Boundaries to the search space.
        """
        super().__init__(file_name, bounds=bounds, seed=seed)
        self.device = device
        
    def _generate_random_candidate(self) -> None:
        r"""Uniformly generate a candidate in the known boundaries"""
        candidate = {}
        for bound_key in self.bounds:
            min_bound, max_bound = self.bounds[bound_key]
            # Uniformly choose a number from the designated range
            param = np.random.uniform(min_bound, max_bound)
            candidate[bound_key] = param
        return candidate
                
    def _initialize_model(self, candidate, training_result, state_dict=None):
        r""" TODO

        :param candidate: 
        """
        # Group candidates, objectives and variances from observations 
        train_x, train_obj, train_var = [], [], []
        for o in self.observations:
            train_x.append(list(o["candidate"].values()))
            train_obj.append(o["result"][0])
            train_var.append(o["result"][1])
        train_x.append(list(candidate.values()))
        train_obj.append(training_result[0])
        train_var.append(training_result[1])
        
        # Put into torch tensor
        train_x = torch.tensor(train_x, device=self.device, dtype=NEIOptimizer.DTYPE)
        train_obj = torch.tensor(train_obj, device=self.device, dtype=NEIOptimizer.DTYPE).unsqueeze(-1)
        train_var = torch.tensor(train_var, device=self.device, dtype=NEIOptimizer.DTYPE).unsqueeze(-1)

        # define models for objective and constraint
        model = HeteroskedasticSingleTaskGP(train_x, train_obj, train_var).to(train_x)
        # combine into a multi-output GP model
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # load state dict if it is passed
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll, model
        
    # TODO: Change model initialization upon NanError thrown by cholesky decomposition
    def generate_candidate(self, 
                           candidate: Union[Dict[str, Any], None],
                           trainer_info: Union[Dict, None]) \
            -> Dict[str, Any]:
        if candidate is None or trainer_info is None and len(self.observations) == 0:
            # Generate a random candidate to startup the optimizing process
            return self._generate_random_candidate()
        elif candidate is None or trainer_info is None and len(self.observations) > 0:
            # TODO: Need to handle this case
            pass
        else:
            print(f"Optimizer received \nCandidate: {candidate} \nTrainer info: {trainer_info}")
            mll, model = self._initialize_model(candidate, trainer_info["result"])
            fit_gpytorch_model(mll)
            qmc_sampler = SobolQMCNormalSampler(num_samples=NEIOptimizer.MC_SAMPLES, seed=self.seed)
            qNEI = qNoisyExpectedImprovement(
                model=model, 
                X_baseline=torch.tensor([list(o["candidate"].values()) for o in self.observations],
                                        device=self.device, dtype=NEIOptimizer.DTYPE),
                sampler=qmc_sampler, 
            )

            # Torch based bounds
            lower_bounds = [bound[0] for bound in self.bounds.values()]
            upper_bounds = [bound[1] for bound in self.bounds.values()]
            print(f"Bounds: {lower_bounds}, {upper_bounds}")
            bounds_torch = torch.tensor([lower_bounds, upper_bounds], device=self.device, dtype=NEIOptimizer.DTYPE)
            
            # Try-except to handle a weird bug
            # 
            try:
                torch_candidate, _ = optimize_acqf(
                    acq_function=qNEI,
                    bounds=bounds_torch,
                    q=1,              # Generate only one candidate at a time
                    num_restarts=10,  # ???
                    raw_samples=500,  # Sample on GP using Sobel sequence
                    options={
                        "batch_limit": 5,
                        "max_iter": 200,
                        "seed": self.seed
                    }
                )  
            except NanError as e:
                print("The weird bug showed up. Using another candidate..")
                return self.generate_candidate(candidate, trainer_info)
            
            candidate = {}
            for i, key in enumerate(self.get_labels()):
                candidate[key] = torch_candidate.cpu().numpy()[0][i]
                
            print(f"Got: {trainer_info}, sending candidate: {candidate}\n"
                  f"Number of observations: {len(self.observations)}, "
                  f"Number of Trainers running: {self.num_trainers}")
                
            return candidate
                
            
            