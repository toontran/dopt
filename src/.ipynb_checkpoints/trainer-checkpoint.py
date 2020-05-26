import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any
import json


class Trainer(ABC):
    r"""Abstract base class for model trainers."""

    def __init__(self, host="127.0.0.1", port="15555"):
        r""" TODO

        :param host:
        :param port:
        """
        self.trainer_info = {}
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
            # Receive the TODO
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
        r""" TODO

        :param candidate:
        :return:
        """
        raise NotImplementedError
