import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Tuple, Optional
import json
import math
from os import path
import random

import torch
import numpy as np

from dopt.utils import generate_seed

# TODO: Use logging instead of print
# Multiple objective functions?
# TODO: Deal with ordering problem
class Optimizer(ABC):
    r"""Abstract base class for hyperparameter optimizers.
    Optimizer distributes candidates (sets of hyperparameters)
    to Trainers, each of which is on a different machine to 
    compute the objective function in parallel.
    """
    MAX_OBSERVATIONS = 500

    def __init__(self, 
                 file_name: str,
                 bounds: Dict[str, Tuple[float, float]],
                 seed: Optional[int] = random.randint(1, 100000)) -> None:
        r""" Constructor for Optimizer base class.
        
        :param file_name: Name of the file
                          that stores observations
        :param bounds:    Boundaries to the search space
        """
        self.file_name = file_name
        self.trainers = []
        self.bounds = bounds
        # Ensure reproducibility
        random.seed(seed)
        torch.manual_seed(generate_seed())
        np.random.seed(generate_seed())
        # List of observed points:
        # [{"candidate":..., "result":...}, ...}]
        self.observations: List[Dict[str, Dict]] = []
        self._load_observations()
        # List of pending hyperparameters, length = number of Trainers
        # [{"num_batch":..., "num_iter":...}, ...]
        self.pending_candidates: List[Dict[str, Dict]] = []
        # The server loop
        self.loop = asyncio.get_event_loop()
            
    def is_running(self) -> bool:
        r"""Determine where the optimizer will stop. Override the
        function to add stopping condition."""
        if len(self.observations) > Optimizer.MAX_OBSERVATIONS:
            return False
        return True
    
    def get_labels(self):
        return self.bounds.keys()
    
    def _load_observations(self):
        r"""Load observations from existing file. If file doesn't
        exist, create a new file"""
        if path.exists(self.file_name):
            # Load observations
            with open(self.file_name, "r") as f:
                for line in f.readlines():
                    self.observations.append(json.loads(line))
        else:
            # Create a new file
            with open(self.file_name, "w") as f:
                pass
    
    def _save_observation(self, observation):
        r"""Save the acquired observation into a storing file"""
        with open(self.file_name, "a") as f:
            f.write(json.dumps(observation, indent=None) + "\n")
            
    def get_best_observation(self, scorer):
        r"""Return highest score given by scorer, which
        is a function that takes in the objective value and variance
        
        :param scorer: A function that takes in the objective value and variance as input
        :return:       The best observation according to the scorer
        """
        highest_score = -math.inf
        best_observation = None
        for observation in observations:
            obj_value, obj_var = observation["result"]
            score = scorer(obj_value, obj_var)
            if score > best_observation:
                highest_score = score
                best_observation = observation
        return best_observation

    def run(self, host="127.0.0.1", port="15555") -> None:
        """ Runs server at specified host and port.

        :param host: TODO
        :param port:
        """
        asyncio.run(self._start_server(host, port))

    async def _start_server(self, host, port) -> None:
        server = await asyncio.start_server(self._handle_trainer,
                                            host, port)
        address = server.sockets[0].getsockname()
        print(f'Serving on {address}')
        async with server:
            await server.serve_forever()
        print("Done")

    async def _handle_trainer(self, reader: asyncio.StreamReader,
                              writer: asyncio.StreamWriter) -> None:
        r"""Handle a single Trainer. Receive incoming candidate request
        and send one potential candidate to the Trainer
        
        Inner working mechanism: TODO

        :param reader: TODO
        :param writer:
        """
        print(f"Connected with Trainer at "
              f"{writer.get_extra_info('peername')}")
        
        
        # Add an empty slot to accomodate the pending candidate from the Trainer
        trainer_ip = writer.get_extra_info('peername')[0]
        self.trainers.append(trainer_ip)
        self.pending_candidates.append(None) 
        
        trainer_index = len(self.trainers) - 1
        trainer_info = None
        candidate = None
        while self.is_running():
            
            # Find one potential candidate to try next based on the info
            candidate: Dict[str, Any] = self.generate_candidate()
            candidate["ip"] = trainer_ip
            
            # Send candidate to Trainer
            out_message = json.dumps(candidate)
            writer.write(out_message.encode("utf8"))
            await writer.drain()
            
            candidate.pop("ip") # We don't really need ip though..
            self.pending_candidates[trainer_index] = candidate
            
            # Receive info of the Trainer including training result(s)
            try:
                in_message: str = (await reader.read(1023)).decode("utf8")
                trainer_info: Dict = json.loads(in_message)
            except ValueError:
                print(f"Error in receiving info: {trainer_info}, shutting down connection with Trainer {trainer_ip}")
                break
                
            self.handle_info_received(trainer_ip, trainer_index, candidate, trainer_info)
            
        print(f"Closing Trainer at {writer.get_extra_info('peername')}")
        self.pending_candidates[trainer_index] = None
        writer.close()
        
    def handle_info_received(self, 
                           trainer_ip: str,
                           trainer_index: int,
                           candidate: Dict[str, Any], 
                           trainer_info: Dict) -> None:
        r"""Puts together information received into an observation.
        
        :param trainer_index: ID of the trainer the info comes from.
        :param candidate:     Candidate being used.
        :param trainer_info: The information the trainer gives: 
                             the training results, running time, etc.
        """
        observation = {
            "ip": trainer_ip,
            "candidate": candidate, 
            "result": trainer_info["result"],
            "time_started": trainer_info["time_started"],
            "time_elapsed": trainer_info["time_elapsed"]
        }
        self.observations.append(observation)
        self.pending_candidates[trainer_index] = None
        self._save_observation(observation)

    @abstractmethod
    def generate_candidate(self) \
            -> Dict[str, Any]:
        r"""Draw the best candidate to evaluate based on known observations."""
        raise NotImplementedError
            
