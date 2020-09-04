import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional,\
                   Tuple, Callable
import json
import random
from time import sleep
import logging

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
from botorch.utils.sampling import draw_sobol_normal_samples
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform
)

from dopt.optimizers.optimizer import Optimizer
from dopt.utils import generate_seed


class qNEIModified(qNoisyExpectedImprovement):
    def __init__(self, model: Model, X_baseline: Tensor,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_pending: Optional[Tensor] = None,
        prune_baseline: bool = False,
    ) -> None:
        super().__init__(model=model, X_baseline=X_baseline, sampler=sampler, objective=objective,
                         X_pending=X_pending, prune_baseline=prune_baseline)
        self.count = 0
        
        
    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qNoisyExpectedImprovement on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Noisy Expected Improvement values at the
            given design points `X`, where `batch_shape'` is the broadcasted batch shape
            of model and input `X`.
        """
        self.count += 1
        q = X.shape[-2]
        match_batch_shape(self.X_baseline, X)
        X_full = torch.cat([X, match_batch_shape(self.X_baseline, X)], dim=-2)        
        self.posterior = self.model.posterior(X_full)
        samples = self.sampler(self.posterior)
        obj = self.objective(samples)
        diffs = obj[:, :, :q].max(dim=-1)[0] - obj[:, :, q:].max(dim=-1)[0]
        return diffs.clamp_min(0).mean(dim=0)


# Find mean and variance of Gaussian Process
# TODO: Change this to a Noisy Constrained Optimizer, and acquisition
#       function can be changed from input
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
            initial_candidates: Optional[list] = [],
            device: Optional[str] = "cpu",
            seed: Optional[int] = random.randint(1, 100000),
        ) -> None:
        r"""Constructor for  Bayesian optimizer that use Noisy Expected Improvement
        as the acquisition function. 
        
        :param device:              Generate candidates on the chosen device.
        :param bounds:              Boundaries to the search space.
        """
        super().__init__(file_name, bounds=bounds, seed=seed)
        self.device = device
        self.current_model = None
        self.num_constraints = None
        self.initial_candidates = initial_candidates
        
    def _generate_random_candidate(self) -> Dict[str, Any]:
        r"""Randomly generate a candidate in the known boundaries. Is uniform random."""
        candidate = {}
        for bound_key in self.bounds:
            min_bound, max_bound = self.bounds[bound_key]
            # Uniformly choose a number from the designated range
            param = np.random.uniform(min_bound, max_bound)
            candidate[bound_key] = param
        return candidate
    
    def _get_observation_data(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Puts known observations into appropriate tensors
        to be passed into the predictive model.
        """
        num_constraints_received = len(self.observations[-1]["constraints"])
        assert self.num_constraints == None or \
            num_constraints_received == self.num_constraints, \
                f"Inapproriate number of constraints, {num_constraints_received}"\
                f" constraint(s) returned, while {self.num_constraints} constraint(s) allowed"
        if self.num_constraints == None:
            self.num_constraints = len(self.observations[-1]["constraints"])
            
        # Group candidates, objectives, variances and constraints from observations 
        train_x, train_obj, train_var= [], [], []
        train_cons = [[] for i in range(self.num_constraints)]
        for o in self.observations:
            train_x.append(list(o["candidate"].values())[:-1]) # Last key of candidate is id
            train_obj.append(o["objective"][0])
            train_var.append(o["objective"][1])
            for i in range(self.num_constraints):
                train_cons[i].append(o["constraints"][i])
                
        # Put into torch tensor 
        train_x = torch.tensor(train_x, device=self.device, dtype=NEIOptimizer.DTYPE)
        train_obj = torch.tensor(train_obj, device=self.device, dtype=NEIOptimizer.DTYPE).unsqueeze(-1)
        train_var = torch.tensor(train_var, device=self.device, dtype=NEIOptimizer.DTYPE).unsqueeze(-1)
        train_cons = torch.tensor(train_cons, device=self.device, dtype=NEIOptimizer.DTYPE).unsqueeze(-1)  
        return train_x, train_obj, train_var, train_cons
                
    def _initialize_model(self, state_dict: Optional[Dict] = None):
        r""" Create the model that predicts values of candidate and 
        load state dict (if available).

        :param state_dict: State of the previous model (fitting model is
                           easier/faster when specified)
        """
        train_x, train_obj, train_var, train_cons = self._get_observation_data()      

        # define models for objective and constraint
        model_obj = HeteroskedasticSingleTaskGP(train_x, train_obj, train_var).to(train_x)
        
        model_cons = []
        for i in range(self.num_constraints):
            model_cons.append(
                FixedNoiseGP(train_x, train_cons[i], 
                             torch.tensor(0.1, device=self.device).expand_as(train_cons[i]))\
                                    .to(train_x)
            )
            
        # combine into a multi-output GP model
        model = ModelListGP(model_obj, *model_cons) 
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        # load state dict if it is passed
        if state_dict is not None:
            model.load_state_dict(state_dict)
        elif self.current_model is not None:
            model.load_state_dict(self.current_model.state_dict())
            
        self.current_model = model
        return mll, model
    
    def _initialize_acqf(self):
        qmc_sampler = SobolQMCNormalSampler(num_samples=NEIOptimizer.MC_SAMPLES, seed=generate_seed())

        # define a feasibility-weighted objective for optimization
        constrained_obj = None
        if self.num_constraints > 0:
            constraint_functions = []
            for i in range(self.num_constraints):
                constraint_idx = i + 1
                print("Constraint index: ", constraint_idx)
                constraint_functions.append(lambda Z: Z[..., constraint_idx])
            constrained_obj = ConstrainedMCObjective(
                objective=lambda Z: Z[..., 0],
                constraints=constraint_functions
            )
        self.qNEI = qNEIModified(
            model=self.current_model, 
            X_baseline=self.observation_to_candidate_tensor(),
            X_pending=self.pending_candidate_list_to_tensor(),
            sampler=qmc_sampler,
            objective=constrained_obj
        )
        
    def generate_candidate(self) -> Dict[str, Any]:
        r"""Draw the best candidate to evaluate based on known observations.
        
        :param is_random: Set to True if want to generate randomly
        """
        
        if len(self.initial_candidates) > 0:
            candidate = self.initial_candidates.pop()
            candidate = dict(sorted(candidate.items()))
            candidate["id"] = self.generate_id()
            return candidate
        elif len(self.observations) == 0 and len(self.initial_candidates) == 0:
            # If no previous observation or if initial candidate(s) are specified (may be more than one)
            # Then use initial candidate(s), If no initial candidate available, generate randomly.
            print("Generating initial candidate(s)")
            return self._generate_random_candidate()

        mll, model = self._initialize_model()
        fit_gpytorch_model(mll)
        self._initialize_acqf()

        # Turn dictionary bounds into torch bounds
        lower_bounds = [bound[0] for bound in self.bounds.values()]
        upper_bounds = [bound[1] for bound in self.bounds.values()]
        bounds_torch = torch.tensor([lower_bounds, upper_bounds], device=self.device, dtype=NEIOptimizer.DTYPE)

        # Try-except to handle a weird bug
        try:
            torch_candidate, _ = optimize_acqf(
                acq_function=self.qNEI,
                bounds=bounds_torch,
                q=1,              # Generate only one candidate at a time
                num_restarts=10,  # ???
                raw_samples=500,  # Sample on GP using Sobel sequence
                options={
                    "batch_limit": 5,
                    "max_iter": 200,
                    "seed": generate_seed()
                }
            )  
        except NanError as e:
            raise Exception("NanError again. There's no way to solve it yet !!!")

        # Put parameters together
        candidate = {}
        for i, key in enumerate(self.get_labels()):
            candidate[key] = torch_candidate.cpu().numpy()[0][i]

        return candidate
        
    def observation_to_candidate_tensor(self):
        return torch.tensor([list(o["candidate"].values())[:-1] for o in self.observations], 
                     device=self.device, dtype=NEIOptimizer.DTYPE)

    def pending_candidate_list_to_tensor(self):
        t = torch.tensor([list(c.values())[:-1] for c in self.pending_candidates], 
                     device=self.device, dtype=NEIOptimizer.DTYPE)
        if t.shape[-1] == 0:
            return None
        print("Pending shape:", t.shape)
        return t
                
            
            