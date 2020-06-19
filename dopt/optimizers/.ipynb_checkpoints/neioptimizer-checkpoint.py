import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional,\
                   Tuple, Callable
import json
import random
from time import sleep

import torch
from torch import Tensor
import numpy as np
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.utils.errors import NanError

from botorch import fit_gpytorch_model
from botorch.models import HeteroskedasticSingleTaskGP, FixedNoiseGP, ModelListGP
from botorch.models.model import Model
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.objective import IdentityMCObjective, MCAcquisitionObjective
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import ConstrainedMCObjective

from dopt.optimizers.optimizer import Optimizer


# Find mean and variance of Gaussian Process
class NEIOptimizer(Optimizer):
    r"""A Bayesian Optimizer that uses Noisy Expected Improvement
    as the acquisition function.
    
    Example:
        >>> bounds = TODO
        >>> filename = TODO
        >>> optimizer = NEIOptimizer(filename, bounds, device="cuda:0")
        >>> optimizer.run()
    """
    MC_SAMPLES = 500
    DTYPE = torch.double
    
    def __init__(
            self, 
            file_name: str,
            bounds: Dict[str, Tuple[float, float]],
            device: Optional[str] = "cpu",
            seed: Optional[int] = random.randint(1, 100000),
            get_feasibility: Optional[Callable[[Tensor], float]] = None,
            initial_candidate: Optional[Union[
                Dict[str, Any], # A single candidate to evaluate
                List[Dict[str, Any]], # Multiple candidates to evaluate
                None]] = None
        ) -> None:
        r"""Constructor for  Bayesian optimizer that use Noisy Expected Improvement
        as the acquisition function. 
        
        :param device:              Generate candidates on the chosen device.
        :param bounds:              Boundaries to the search space.
        :param get_feasibility: L1 constraint of user choice. The satisfiability of the candidate is 
                                       taken into account when choosing the best candidate, i.e. avoid
                                       infeasible candidates. Feasible when the returned value is <= 0.
        """
        super().__init__(file_name, bounds=bounds, seed=seed)
        self.device = device
        self.get_feasibility = lambda x: 0 if get_feasibility == None else get_feasibility(x)
        self.initial_candidate = initial_candidate
        self.current_model = None
        
    # TODO: Handle Trainer changing the candidate themselves (have Trainers send candidate also?)
    def handle_observation(self, 
                           trainer_index: int,
                           candidate: Dict[str, Any], 
                           trainer_info: Dict) -> None:
        r"""Puts together information received into an observation.
        
        :param trainer_index: ID of the trainer the info comes from.
        :param candidate:     Candidate being used.
        :param trainer_info: The information the trainer gives: 
                             the training results, running time, etc.
        """
        candidate_tensor = torch.tensor(list(candidate.values()), 
                                        device=self.device)
        observation = {
            "id": trainer_index,
            "candidate": candidate, 
            "result": trainer_info["result"],
            "time_started": trainer_info["time_started"],
            "time_elapsed": trainer_info["time_elapsed"],
            "feasibility": self.get_feasibility(candidate_tensor)
        }
        self.observations.append(observation)
        self.pending_candidates[trainer_index] = None
        self._save_observation(observation)
        
    def _generate_random_candidate(self) -> Dict[str, Any]:
        r"""Uniformly generate a candidate in the known boundaries"""
        candidate = {}
        for bound_key in self.bounds:
            min_bound, max_bound = self.bounds[bound_key]
            # Uniformly choose a number from the designated range
            param = np.random.uniform(min_bound, max_bound)
            candidate[bound_key] = param
        if self.get_feasibility(
            torch.tensor(list(candidate.values()), device=self.device)
        ) > 0:
            # TODO: Generating random candidate may not result in a feasible one
            #       or do so in great amount of time, need to put Warnings
            return self._generate_random_candidate() # Random until feasible
        print("Feasible!")
        return candidate
    
    def _get_observation_data(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Puts known observations into appropriate tensors
        to be passed into the predictive model"""
        # Group candidates, objectives and variances from observations 
        train_x, train_obj, train_var, train_con = [], [], [], []
        for o in self.observations:
            train_x.append(list(o["candidate"].values()))
            train_obj.append(o["result"][0])
            train_var.append(o["result"][1])
            train_con.append(o["feasibility"])
        # Put into torch tensor 
        train_x = torch.tensor(train_x, device=self.device, dtype=NEIOptimizer.DTYPE)
        train_obj = torch.tensor(train_obj, device=self.device, dtype=NEIOptimizer.DTYPE).unsqueeze(-1)
        train_var = torch.tensor(train_var, device=self.device, dtype=NEIOptimizer.DTYPE).unsqueeze(-1)
        train_con = torch.tensor(train_con, device=self.device, dtype=NEIOptimizer.DTYPE).unsqueeze(-1)  
        return train_x, train_obj, train_var, train_con
                
    def _initialize_model(self, state_dict: Optional[Dict] = None):
        r""" Create the model that predicts values of candidate and 
        load state dict (if available).

        :param state_dict: State of the previous model (fitting model
                           easier/faster when specified)
        """
        train_x, train_obj, train_var, train_con = self._get_observation_data()      

        # define models for objective and constraint
        model_obj = HeteroskedasticSingleTaskGP(train_x, train_obj, train_var).to(train_x)
        model_con = FixedNoiseGP(train_x, train_con, 
                                 torch.tensor(0.0, device=self.device).expand_as(train_con))\
                        .to(train_x)
        # combine into a multi-output GP model
        model = ModelListGP(model_obj, model_con) 
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        # load state dict if it is passed
        if state_dict is not None:
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(self.current_model.state_dict())
        return mll, model
        
    def generate_candidate(self) \
            -> Dict[str, Any]:
        r"""Draw the best candidate to evaluate based on known observations."""
        
        if len(self.observations) == 0 or self.initial_candidate != None:
            # If no previous observation or if initial candidate(s) are specified (may be more than one)
            # Then use initial candidate(s), If no initial candidate available, generate randomly.
            print("Generating initial candidate")
            if self.initial_candidate == None:
                return self._generate_random_candidate()
            elif isinstance(self.initial_candidate, list) and len(self.initial_candidate) > 0:
                initial_candidate = self.initial_candidate.pop()
            elif not self.initial_candidate:
                self.initial_candidate = None
                return self.generate_candidate()
            else:
                initial_candidate = self.initial_candidate
            
            if self.get_feasibility(
                torch.tensor(list(initial_candidate.values()), device=self.device)
            ) > 0:
                raise Exception("User initial candidate is not feasible!")
            else:
                return initial_candidate
        else:
            print(f"Optimizer received \n{self.observations[-1]}")
            mll, model = self._initialize_model()
            fit_gpytorch_model(mll)
            self.current_model = model
            qmc_sampler = SobolQMCNormalSampler(num_samples=NEIOptimizer.MC_SAMPLES, seed=self.seed)

            # define a feasibility-weighted objective for optimization
            constrained_obj = ConstrainedMCObjective(
                objective=lambda Z: Z[..., 0],
                constraints=[lambda Z: Z[..., 1]],
            )
            qNEI = qNEIModified(
                model=model, 
                X_baseline=torch.tensor([list(o["candidate"].values()) for o in self.observations],
                                        device=self.device, dtype=NEIOptimizer.DTYPE),
                sampler=qmc_sampler,
                objective=constrained_obj
            )

            # Turn dictionary bounds into torch bounds
            lower_bounds = [bound[0] for bound in self.bounds.values()]
            upper_bounds = [bound[1] for bound in self.bounds.values()]
            bounds_torch = torch.tensor([lower_bounds, upper_bounds], device=self.device, dtype=NEIOptimizer.DTYPE)
            
            # Try-except to handle a weird bug
            count = 3
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
                self.seed = self.seed + 1
                count -= 1
                if count <= 0:
                    raise Exception("WEIRDD")
                return self.generate_candidate()
            
            # Put parameters together
            candidate = {}
            for i, key in enumerate(self.get_labels()):
                candidate[key] = torch_candidate.cpu().numpy()[0][i]
                
            print(f"Sending candidate: {candidate}\n"
                  f"Number of observations: {len(self.observations)}, "
                  f"Number of Trainers running: {self.num_trainers}")
                
            return candidate
                
            
            