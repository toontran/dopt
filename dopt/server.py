import os, sys, time
from typing import Dict, List, Union
import json
import logging

import socket
from _thread import start_new_thread
from threading import Lock

from multiprocessing import Pipe, Process, Queue, Manager
from multiprocessing.connection import Connection
# from queue import SimpleQueue

from dopt import Optimizer
from dopt.utils import process_commands_in_parallel


class Server:
    
    def __init__(self, 
                 optimizer: Optimizer,
                 config: Dict,
#                  initial_candidates: Union[None, List[Dict]] = None,
                 logging_level = logging.ERROR
        ) -> None:
        """Need docs on the config"""
        self.optimizer = optimizer
        self.process_list = []
        self.config = config
        self.trainers = Manager().dict()
        self.trainer_id = 0
        self.trainer_queue = Queue()
        self.bad_candidate_queue = Queue()
#         self.initial_candidates = initial_candidates \
#                         if isinstance(initial_candidates, list) else []
        self.logging_level = logging_level
        self.server_logger = self.init_log(stdout_level=self.logging_level)
        
        # Locks for multiprocess or multithreaded access to resource
        self.lock_trainers = Lock()
        self.lock_trainer_queue = Lock()
        self.lock_optimizer_conn = Lock()
        self.lock_server_logger = Lock()
        self.lock_bad_candidate_queue = Lock()
        
    def run(self):
        """Starts 3 main Processes: One for the Optimizer, one for
        starting trainers, one for connecting with trainers."""
        
        # Runs the Optimizer
        self.optimizer_conn, cconn = Pipe()
        optimizer_process = Process(target=self.optimizer.run, 
                                    args=(cconn, self.server_logger, self.lock_server_logger))
        optimizer_process.daemon = True
        self.process_list.append(optimizer_process)
        
        # Establish server connections, waits for trainers to connect
        listen_trainers_process = Process(target=self.listen_trainers, args=())
        listen_trainers_process.daemon = True
        self.process_list.append(listen_trainers_process)
        
        # Startup trainers on target remote machines
        startup_process = Process(target=self.startup_trainers, args=())
        startup_process.daemon = False # This process spawns children processes
        self.process_list.append(startup_process)
        
        # Start all processes
        for p in self.process_list:
            p.start()
        
        while True:
            with self.lock_server_logger:
                with self.lock_trainers:
                    self.server_logger.debug(f"Number of Trainers running: {len(self.trainers)}")
                with self.lock_trainer_queue:
                    self.server_logger.debug(f"Number of Trainers in the Queue: {self.trainer_queue.qsize()}")
            if not self.trainer_queue.empty():
                with self.lock_server_logger:
                    self.server_logger.debug("A Trainer is ready")
                    
                # Check which Trainer crashed, then push into a new Queue; if Trainer from 
                candidate = None
                with self.lock_optimizer_conn:
                    with self.lock_bad_candidate_queue:
                        if not self.bad_candidate_queue.empty():
                            candidate = self._dequeue_bad_candidate()
                        elif self.optimizer_conn.poll(None): # There's a candidate available
                            message = self.optimizer_conn.recv()
                            message = json.loads(message)
                            candidate = message["candidate"]
                            
                        if candidate != None:
                            connection, address, is_active, pending_candidate, trainer_id = self._dequeue_trainer()                    
                            self._send_candidate_to_trainer(candidate, connection, address)        
                            with self.lock_trainers:
                                self.trainers[trainer_id] = [*self.trainers[trainer_id][:3], candidate] 

                            with self.lock_trainers:
                                with self.lock_server_logger:
                                    self.server_logger.debug(f"Trainers running: {json.dumps({trainer_id: self.trainers[trainer_id][1:] for trainer_id in list(self.trainers)})}, assigning {candidate} to {trainer_id}.")
            else:
                pass
                            

            if not optimizer_process.is_alive():
                with self.lock_server_logger:
                    self.server_logger.debug("Optimizer stopped. Killing all processes.")
                break
                
            time.sleep(1)
                
        self.terminate()
        
    def terminate(self):
        """Kill all Processes"""
        for p in self.process_list:
            p.kill()
            
    def _remove_pending_candidate(self, pending_candidate):
        """Tells the Optimizer to drop candidate off pending list"""
        with self.lock_server_logger:
            self.server_logger.warning(f"Removing candidate: {pending_candidate}")
        with self.lock_optimizer_conn:
            self.optimizer_conn.send(Optimizer.HEADER_REMOVE_CANDIDATE + \
                                     json.dumps(pending_candidate)+'\n')
            
    def _dequeue_bad_candidate(self):
        with self.lock_bad_candidate_queue:
            trainer = self.bad_candidate_queue.get() # Block until found one
        return trainer
            
    def _dequeue_trainer(self):
        """Dequeues one trainer from the queue, return trainer info."""
        with self.lock_trainer_queue:
            trainer_id = self.trainer_queue.get() # Block until found one
        with self.lock_trainers:
            if trainer_id not in self.trainers:
                return self._dequeue_trainer()
            
            pending_candidate = None
            if len(self.trainers[trainer_id]) == 3:
                connection, address, is_active = self.trainers[trainer_id]
            elif len(self.trainers[trainer_id]) == 4:
                connection, address, is_active, pending_candidate = self.trainers[trainer_id]
            else:
                raise Exception(f"self.trainers contains wrong things: {self.trainers[trainer_id]}")
        
            if is_active == 0 and pending_candidate:
                # Remove corrupted Trainer & dequeue again
                self._remove_pending_candidate(pending_candidate)
                self.trainers.pop(trainer_id)
                self.__dequeue_trainer()
        return connection, address, is_active, pending_candidate, trainer_id
            
    def _send_candidate_to_trainer(self, candidate, connection, address):
        """Send a candidate safely to a Trainer on the queue"""
        try:
            connection.sendall(str.encode(
                json.dumps({"candidate": candidate}) + "\n"
            ))
            with self.lock_server_logger:
                self.server_logger.debug(json.dumps({'candidate_sent':candidate, 'address':address}))
        except Exception as e:
            with self.lock_server_logger:
                self.server_logger.exception(f"Problem with address: {address}")
    
    def startup_trainers(self):
        """Runs on another Process. SSH into each machine in the list,
        and start the Trainers with commands specified."""
        commands = []
        for host_cat in self.config["computer_list"]:
            for host in self.config["computer_list"][host_cat]:
                commands.append({
                    "host": host, 
                    "command": self.config["commands"][host_cat]
                })
        with self.lock_server_logger:
            self.server_logger.debug("Starting trainers..")
        process_commands_in_parallel(commands)
        
    def listen_trainers(self):
        """Runs on another Process. Spawns threads to handle communication
        with Trainers."""
        server = socket.socket()
        host = '0.0.0.0'
        port = self.config["server"]["port"]
        try:
            server.bind((host, port))
        except socket.error as e:
            with self.lock_server_logger:
                self.server_logger.exception("Connection Error")

        with self.lock_server_logger:
            self.server_logger.debug('Waiting for a Connection..')
        server.listen(5)
        while True:
            client, address = server.accept()
            with self.lock_server_logger:
                self.server_logger.debug('Connected to: ' + address[0] + ':' + str(address[1]))
            start_new_thread(self.threaded_client_handling, (client, address,))
        
    def threaded_client_handling(self, connection, address):
        """A thread handling the Trainers. Continually communicate 
        with the Trainers to gather real-time info on the Trainers."""
        with self.lock_trainers:
            # Quick fix for multiple Trainer instances running on same machine
            if len(self.trainers) > 0:
                for trainer_id in self.trainers.keys():
                    if address[0] == self.trainers[trainer_id][1][0]: 
                        return
            self.trainer_id += 1
            self.trainers[self.trainer_id] = [connection, address, 1] # 1 means active
        trainer_id = self.trainer_id
        while True:
            # Receive message from trainers
            try:
                responses = connection.recv(10000)
            except Exception as e:
                with self.lock_server_logger:
                    self.server_logger.exception("Can't receive response")
                break
                
            # If trainer exits
            if not responses:
                break

            # Handle message received
            reply = self._handle_client_response(responses, trainer_id, address)

            # Reply back to trainers
            try:
                connection.sendall(str.encode(reply+'\n'))
            except Exception as e:
                with self.lock_server_logger:
                    self.server_logger.exception("Can't send reply")
                break
            # Delay response
            time.sleep(0.5) 
                
        connection.close()
        with self.lock_server_logger:
            self.server_logger.warning(f"Closed connection with {address}")
            
        with self.lock_trainers:
            # Remove corrupted Trainer & dequeue again
            if len(self.trainers[trainer_id]) == 4:
                _, _, status, pending_candidate = self.trainers[trainer_id]
                if status == 2:
                    # Candidate crashes when evaluated (twice in a row)
                    self._remove_pending_candidate(pending_candidate)
                    with self.lock_server_logger:
                        self.server_logger.error("Trainer crashes while evaluating candidate: " + \
                                                 f"{json.dumps(pending_candidate)}")
                elif status == 1:
                    # Trainer crashed: Save candidate to try on a different Trainer
                    with self.lock_bad_candidate_queue:
                        self.bad_candidate_queue.put(pending_candidate)
            elif len(self.trainers[trainer_id]) == 3:
                with self.lock_server_logger:
                    self.server_logger.warning(f"Trainers running: {json.dumps({trainer_id: self.trainers[trainer_id][1:] for trainer_id in self.trainers.keys()})}, Current {trainer_id} has {self.trainers[trainer_id]}.")
            else:
                raise Exception(f"self.trainers contains wrong things: {self.trainers[trainer_id]}")
            self.trainers.pop(trainer_id) 
#             self.trainers[trainer_id][3] = 0 # Trainer not active anymore

    def _handle_client_response(self, responses, trainer_id, address):
        """Handles the response from the Trainers
        
        :param response: Client response contains a dictionary with 
                         following keys:
                         "gpu_info": Information on the machine GPU, CPU, etc.
                         "logging": Using print() on Trainer side will be channeled
                                  into this to show up on Server side.
                         "observation": Only appears when an observation is made,
                          contains a dictionary with keys: "candidate", "result",
                          "info".
        :param trainer_id: ID of a Trainer
        :return: A reply to the Trainer
        """
        responses = responses.decode("utf8")
        logger = self.init_log(address=address, stdout_level=self.logging_level)
        
        for response in responses.split("\n")[:-1]:  
            with self.lock_server_logger:
                self.server_logger.debug(f"Loading response: {response}")
            response = json.loads(response)
            if "observation" in response:
                with self.lock_optimizer_conn:
                    self.optimizer_conn.send(json.dumps(response["observation"])+'\n')
                with self.lock_trainer_queue:
                    self.trainer_queue.put(trainer_id)
                with self.lock_trainers:
                    self.trainers[trainer_id][2] = 1 
                with self.lock_server_logger:
                    self.server_logger.debug(json.dumps(response['observation']))
            if "error" in response:
                with self.lock_server_logger:
                    self.server_logger.error(f'{response["error"]}') #
            if "gpu_info" in response:
                with self.lock_server_logger:
                    self.server_logger.debug(json.dumps(response['gpu_info']))
#                 with self.lock_trainers:
#                     self.trainers[trainer_id][2] = response["gpu_info"] # For now
            if "stack_info" in response:
                # Log 
                stringReceived = logging.makeLogRecord(response)
                logger.handle(stringReceived)
                
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
        return json.dumps({"message": "candidate_sent"}) # Just an empty message 
        
    def init_log(self, address=None, stdout_level=logging.DEBUG):
        logger = logging.getLogger("")
        logger.setLevel(logging.DEBUG)
        # create file handler that logs debug and higher level messages
        filename = "logs_" + self.optimizer.filename.split(".")[0] + \
                  f"_{'client' if address else 'server'}.txt"
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(stdout_level)
        # create formatter and add it to the handlers
        name = json.dumps(address) if address else "server"
        formatter = logging.Formatter(f'[{name} - %(asctime)s - %(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(ch)
        logger.addHandler(fh)
        return logger