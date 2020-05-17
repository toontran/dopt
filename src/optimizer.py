import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union
import json


class Optimizer(ABC):
    r"""Abstract base class for hyperparameter optimizers."""

    def __init__(self) -> None:
        r""" Constructor for Optimizer base class."""
        self.loop = asyncio.get_event_loop()
        self.is_running = True
        self.num_trainers = 0
        # List of observed points:
        # [{"hyperparameters":..., "result":...}, ...}]
        self.observations: List[Dict[str, Dict]] = []
        # List of pending sets of hyperparameters
        # [{"num_batch":..., "num_iter":...}, ...]
        self.pending_candidates: List[Dict[str, Any]] = []

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

    async def _handle_trainer(self, reader: asyncio.StreamReader,
                              writer: asyncio.StreamWriter) -> None:
        r"""Receive incoming candidate request
        and send one potential candidate to the Trainer

        :param reader: TODO
        :param writer:
        """
        print(f"Connected with Trainer at "
              f"{writer.get_extra_info('peername')}")
        self.num_trainers += 1
        trainer_info = None
        while self.is_running:
            # Find one potential candidate to try next based on the info
            candidate = self.generate_candidate(trainer_info)
            # Send candidate to Trainer
            out_message = json.dumps(candidate)
            writer.write(out_message.encode("utf8"))
            await writer.drain()
            # Receive info of the Trainer including training result(s)
            in_message: str = (await reader.read(255)).decode("utf8")
            trainer_info: Dict = json.loads(in_message)
        writer.close()
        print(f"Closing Trainer at {writer.get_extra_info('peername')}")

    @abstractmethod
    def generate_candidate(self, trainer_info: Dict) -> Dict[str, Any]:
        r"""Draw the best candidate to evaluate.

        :param trainer_info: Dictionary containing TODO
        """
        raise NotImplementedError


# class NoisyExpectedImprovementOptimizer(Optimizer):
#
#     def generate_candidate(self, trainer_info: Union[Dict, None]) \
#             -> Dict[str, Any]:
#         if trainer_info is None:
#
