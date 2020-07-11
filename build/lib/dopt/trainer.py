import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
from datetime import datetime

from dopt.utils import add_prefix_to_print


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
        self.host = host
        self.port = port
        self.is_running = True

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
            in_message: str = (await reader.read(1023)).decode("utf8")
            candidate: Dict = json.loads(in_message)
            trainer_ip = candidate.pop("ip")
                
            # Pass candidate to objective function and get result(s)
            start = datetime.now()
            try:
                with add_prefix_to_print(trainer_ip):
                    observation = self.get_observation(candidate)
            except Exception as e:
                print(f"Error!!")
                raise e
            elapsed = datetime.now() - start
            
            # Send result back to optimizers
            trainer_info = {}
            trainer_info["result"] = observation
            trainer_info["time_started"] = start.strftime("%m/%d/%Y-%H:%M:%S")
            trainer_info["time_elapsed"] = round(elapsed.seconds/3600, 2) # In hours, rounded to 2nd decimal
            
            out_message = json.dumps(trainer_info)
            writer.write(out_message.encode("utf8"))
            await writer.drain()
            
        writer.close()
        print("Closing")

    @abstractmethod
    def get_observation(self, candidate: Dict[str, Any]) \
            -> Dict[str, Any]:
        r"""Get observation by plugging the candidate into objective function.
        This method is made abstract to easier modify the objective function
        to run on different platforms.

        :param candidate:
        :return:
        """
        raise NotImplementedError