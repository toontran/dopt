import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json

import torch

class Trainer(ABC):
    r"""Representation of the objective function running on the target machine.
    Optimizer distributes candidates (sets of hyperparameters)
    to Trainers, each of which is on a different machine to 
    compute the objective function in parallel.
    """

    def __init__(self, 
                 host: Optional[str] = "127.0.0.1",
                 port: Optional[str] = "15555"):
        r""" TODO

        :param host:
        :param port:
        """
        self.hardware_info: Dict = self._get_hardware_info()
        self.host = host
        self.port = port
        self.is_running = True
        
    def _get_hardware_info(self) -> Dict:
        return {}

    def run(self):
        asyncio.run(self._listen_to_optimizer())

    async def _listen_to_optimizer(self):
        # Open connection with the server
        reader, writer = await asyncio.open_connection(
            self.host, self.port
        )
        print(f"Connected with Optimizer at "
              f"{writer.get_extra_info('peername')}")
        
        while self.is_running:
            
            # Receive the candidate from Optimizer
            in_message: str = (await reader.read(255)).decode("utf8")
            candidate: Dict = json.loads(in_message)
                
            # Train the model and get result(s)
            observation = self.get_observation(candidate)
            
            # Send result back to optimizers
            out_message = json.dumps(observation)
            writer.write(out_message.encode("utf8"))
            await writer.drain()
            
        writer.close()
        print("Closing")

    @abstractmethod
    def get_observation(self, candidate: Dict[str, Any]) \
            -> Dict[str, Any]:
        r""" Get observation by plugging the candidate into objective function.
        This method is made abstract to easier modify the objective function
        to run on different platforms.

        :param candidate:
        :return:
        """
        raise NotImplementedError