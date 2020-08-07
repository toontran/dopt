import os, sys, time
from typing import Dict, List, Union
import json

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
                 verbose: bool = True
        ) -> None:
        """Need docs on the config"""
        self.optimizer = optimizer
        self.process_list = []
        self.config = config
        self.trainers = Manager().dict()
        self.trainer_id = 0
        self.trainer_queue = Queue()
#         self.initial_candidates = initial_candidates \
#                         if isinstance(initial_candidates, list) else []
        self.verbose = verbose
        # Locks for multiprocess or multithreaded access to resource
        self.lock_trainers = Lock()
        self.lock_trainer_queue = Lock()
        self.lock_optimizer_conn = Lock()
        
    def run(self):
        """Starts 3 main Processes: One for the Optimizer, one for
        starting trainers, one for connecting with trainers."""
        
        # Runs the Optimizer
        self.optimizer_conn, cconn = Pipe()
        optimizer_process = Process(target=self.optimizer.run, args=(cconn,))
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
            if self.verbose:
                with self.lock_trainers:
                    print("Number of Trainers running:", len(self.trainers))
                with self.lock_trainer_queue:
                    print("Number of Trainers in the Queue:", self.trainer_queue.qsize())
            if not self.trainer_queue.empty():
                print("A Trainer is ready")
                
#                 if len(self.initial_candidates) > 0:
#                     candidate = self.initial_candidates.pop()
#                     candidate = dict(sorted(candidate.items())) # Need to sort first
                if self.optimizer_conn.poll(None): # There's a candidate available
                    with self.lock_optimizer_conn:
                        message = self.optimizer_conn.recv()
                    message = json.loads(message)
                    candidate = message["candidate"]
                
                self._send_candidate_to_trainer(candidate)            

            if not optimizer_process.is_alive():
                print("Optimizer stopped. Killing all processes.")
                break
                
            time.sleep(1)
                
        self.terminate()
        
    def terminate(self):
        """Kill all Processes"""
        for p in self.process_list:
            p.kill()
            
    def _send_candidate_to_trainer(self, candidate):
        """Send a candidate safely to a Trainer on the queue"""
        with self.lock_trainer_queue:
            trainer_id = self.trainer_queue.get()
        if trainer_id not in self.trainers:
            return
        connection, address, _ = self.trainers[trainer_id]
        try:
            connection.sendall(str.encode(
                json.dumps({"candidate": candidate}) + "\n"
            ))
            print("Sent candidate")
        except Exception as e:
            print("Problem with address:", address)
            print(e)
    
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
        print("Starting trainers..")
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
            print(str(e))

        print('Waiting for a Connection..')
        server.listen(5)
        while True:
            client, address = server.accept()
            print('Connected to: ' + address[0] + ':' + str(address[1]))
            start_new_thread(self.threaded_client_handling, (client, address,))
        
    def threaded_client_handling(self, connection, address):
        """A thread handling the Trainers. Continually communicate 
        with the Trainers to gather real-time info on the Trainers."""
        with self.lock_trainers:
            self.trainer_id += 1
            self.trainers[self.trainer_id] = [connection, address, None]
        trainer_id = self.trainer_id
        while True:
            # Receive message from trainers
            try:
                responses = connection.recv(10000)
            except Exception as e:
                print("Can't receive response,", e)
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
                print("Can't send reply,", e)
                break
            # Delay response
            time.sleep(0.5) 
                
        connection.close()
        with self.lock_trainers:
            self.trainers.pop(trainer_id)

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
        
        for response in responses.split("\n")[:-1]:  
            if self.verbose:
                print("Loading response: ", response)
            response = json.loads(response)
            if "observation" in response:
                print("Observation found")
                with self.lock_optimizer_conn:
                    self.optimizer_conn.send(json.dumps(response["observation"])+'\n')
                with self.lock_trainer_queue:
                    self.trainer_queue.put(trainer_id)
            if "logging" in response:
                if self.verbose:
                    print(f'[{address}]:{response["logging"]}') # For now
                else:
                    log_file_name = "logs_" + self.optimizer.file_name.split(".")[0] + ".txt"
                    with open(log_file_name, "a") as f:
                        f.write(f"[{address}]:{response['logging']}\n")
            if "gpu_info" in response:
                with self.lock_trainers:
                    self.trainers[trainer_id][2] = response["gpu_info"] # For now
        return json.dumps({"message": "candidate_sent"}) # Just an empty message 
        