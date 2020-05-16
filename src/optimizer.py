import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any
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
        # List of hyperparameters
        self.candidates: List[Dict[str: Any]] = []

    def run(self, host="127.0.0.1", port="15555"):
        asyncio.run(self._start_server(host, port))

    async def _start_server(self, host, port):
        server = await asyncio.start_server(self._handle_trainer,
                                            host, port)
        address = server.sockets[0].getsockname()
        print(f'Serving on {address}')
        async with server:
            await server.serve_forever()

    async def _handle_trainer(self, reader: asyncio.StreamReader,
                              writer: asyncio.StreamWriter):
        r"""Receive incoming candidate request
        and send ONE potential candidate

        :param reader:
        :param writer:
        """
        print(f"Connected with Trainer at "
              f"{writer.get_extra_info('peername')}")
        print(f"Reader {reader}")
        while self.num_observations <= self.max_observations:
            in_message: str = (await reader.read(10)).decode("utf8")
            print("Got dis: ", in_message)
            trainer_info: Dict = json.loads(in_message)
            print(trainer_info)
            candidate = self.generate_candidate(trainer_info)

            out_message = json.dumps(candidate)
            writer.write(out_message.encode("utf8"))
            await writer.drain()
        writer.close()
        print(f"Closing Trainer at {writer.get_extra_info('peername')}")

    @abstractmethod
    def generate_candidate(self, trainer_info: Dict) -> Dict[str, Any]:
        r"""Draw the best candidate to evaluate"""
        raise NotImplementedError
