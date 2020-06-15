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


# TODO: Refactor
class qNEIModified(qNoisyExpectedImprovement):
    r"""A regular Noisy Expected Improvement, but with  """
    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_pending: Optional[Tensor] = None,
        prune_baseline: Optional[bool] = False,
    ) -> None:
        """q-Noisy Expected Improvement.

        Args:
            model: A fitted model.
            X_baseline: A `batch_shape x r x d`-dim Tensor of `r` candidates
                that have already been observed. These points are considered as
                the potential best candidate.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=500, collapse_batch_dims=True)`.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` candidates
                that have points that have been submitted for function evaluation
                but have not yet been evaluated. Concatenated into `X` upon
                forward call. Copied and set to have no gradient.
            prune_baseline: If True, remove points in `X_baseline` that are
                highly unlikely to be the best point. This can significantly
                improve performance and is generally recommended. In order to
                customize pruning parameters, instead manually call
                `botorch.acquisition.utils.prune_inferior_points` on `X_baseline`
                before instantiating the acquisition function.
        """
        super().__init__(
            model=model, X_baseline=X_baseline, sampler=sampler, objective=objective,
            X_pending=X_pending, prune_baseline=prune_baseline
        )
        self.is_possible = lambda x: True if is_possible == None else is_possible(x) 
        
    def prune_impossible_candidates(self, X, result):
        r"""TODO"""
        num_hyperparams = X.shape[-1]
        for i in range(len(result.view(-1, 1))):
            if not self.is_possible(X.view(-1, num_hyperparams)[i]):
                # If not possible, set the lowest score for the candidate
                # TODO: Refactor!!
                result.view(-1, 1)[i] = torch.tensor(float("-inf"), 
                                                                   device=X.device)
                print("N", end="")
            else:
                print("Passed!")
        return result
                
    def forward(self, X: Tensor) -> Tensor:
#         print("Forwarding..", end="")
#         print(X, X.shape)
#         print(self.X_baseline)
        result = super().forward(X)
        return result


# Find mean and variance of Gaussian Process
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
    
    def __init__(
            self, 
            file_name: str,
            bounds: Dict[str, Tuple[float, float]],
            device: Optional[str] = "cpu",
            seed: Optional[int] = random.randint(1, 100000),
            get_feasibility: Optional[Callable[[Tensor], float]] = None,
            initial_candidate: Optional[Union[Dict[str, Any], None]] = None
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
        
    def handle_observation(self, 
                           trainer_index: int,
                           candidate: Dict[str, Any], 
                           trainer_info: Dict) -> None:
        candidate_tensor = torch.tensor(list(candidate.values()), 
                                        device=self.device)
        observation = {
            "id": trainer_index,
            "candidate": candidate, 
            "result": trainer_info["result"],
            "time_started": trainer_info["time_started"],
            "time_elapsed": trainer_info["time_elapsed"],
            "feasibility": self.get_feasibility(candidate_tensor).item()
        }
        self.observations.append(observation)
        self.pending_candidates[trainer_index] = None
        self._save_observation(observation)
        
    def _generate_random_candidate(self) -> None:
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
            return self._generate_random_candidate() # Random until feasible
        print("Feasible!")
        return candidate
                
    def _initialize_model(self, state_dict: Optional[Dict] = None):
        r""" TODO: Refactor variable naming - training_result

        :param candidate: 
        """
        # Group candidates, objectives and variances from observations 
        train_x, train_obj, train_var, train_con = [], [], [], []
        for o in self.observations:
            train_x.append(list(o["candidate"].values()))
            train_obj.append(o["result"][0])
            train_var.append(o["result"][1])
            train_con.append(o["feasibility"])
        
        # Put into torch tensor 
        # TODO: make a util function for turning dict -> torch.tensor
        train_x = torch.tensor(train_x, device=self.device, dtype=NEIOptimizer.DTYPE)
        train_obj = torch.tensor(train_obj, device=self.device, dtype=NEIOptimizer.DTYPE).unsqueeze(-1)
        train_var = torch.tensor(train_var, device=self.device, dtype=NEIOptimizer.DTYPE).unsqueeze(-1)
        train_con = torch.tensor(train_con, device=self.device, dtype=NEIOptimizer.DTYPE).unsqueeze(-1)        

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
        return mll, model
        
    def generate_candidate(self) \
            -> Dict[str, Any]:
        if len(self.observations) == 0:
            # Generate a random candidate to startup the optimizing process
            print("Generating initial candidate")
            if self.initial_candidate == None:
                return self._generate_random_candidate()
            elif self.get_feasibility(
                torch.tensor(list(self.initial_candidate.values()), device=self.device)
            ) > 0:
                raise Exception("User initial candidate is not feasible!")
            else:
                return self.initial_candidate
        else:
            print(f"Optimizer received \n{self.observations[-1]}")
            mll, model = self._initialize_model()
            fit_gpytorch_model(mll)
            qmc_sampler = SobolQMCNormalSampler(num_samples=NEIOptimizer.MC_SAMPLES, seed=self.seed)
            
            print("Fitting...")
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

            # Torch based bounds
            lower_bounds = [bound[0] for bound in self.bounds.values()]
            upper_bounds = [bound[1] for bound in self.bounds.values()]
            bounds_torch = torch.tensor([lower_bounds, upper_bounds], device=self.device, dtype=NEIOptimizer.DTYPE)
            
            # Try-except to handle a weird bug
            print("Finding candidate")
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
                return self.generate_candidate(candidate, trainer_info)
            
            print("Returning result...")
            candidate = {}
            for i, key in enumerate(self.get_labels()):
                candidate[key] = torch_candidate.cpu().numpy()[0][i]
                
            print(f"Sending candidate: {candidate}\n"
                  f"Number of observations: {len(self.observations)}, "
                  f"Number of Trainers running: {self.num_trainers}")
                
            return candidate
                
            
            