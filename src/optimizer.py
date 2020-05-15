import asyncio
from abc import ABC, abstractmethod
from typing import Dict
import json


class Optimizer(ABC):
    r"""Abstract base class for hyper-parameter optimizers"""

    def __init__(self, max_observations: int = -1) -> None:
        r""" Constructor for Optimizer base class

        :param max_observations: Terminates the optimization when the
                                 set maximum number of observations reached

        Example: TODO
        """
        self.loop = asyncio.get_event_loop()
        self.max_observations = max_observations
        self.num_observations = 0

    def run(self, host="localhost", port="15555"):
        self.loop.create_task(asyncio.start_server(
            self.handle_trainer, host, port
        ))

    async def handle_trainer(self, reader, writer):
        trainer_info = {}
        while self.num_observations <= self.max_observations:
            received_message = (await reader.read()).decode("utf8")
            trainer_info = json.loads(received_message)

    @abstractmethod
    def generate_candidate(self):
        r"""Draw the best candidate to evaluate"""
        raise NotImplementedError
