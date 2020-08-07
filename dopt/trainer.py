import time
import os, sys
import json
from typing import Callable, Dict, Tuple, Union
from datetime import datetime

import socket
from multiprocessing import Process, Pipe, Value
from threading import Lock

import torch

from dopt.utils import get_all_gpu_processes_info


NUM_BYTES_RECEIVE = 1024
MAXIMUM_ALLOWED_GPU_PERCENTAGE = 0.9
SERVER_TRAINER_MESSAGE_INTERVAL = 5


class Trainer:
    
    def __init__(self, 
                 objective_function: Callable[[Dict], Tuple],
                 username: str,
                 host: str,
                 port: Union[int, str],
                 num_constraints: int = 0,
                 verbose: bool = True):
        """
        
        :param username: The username we're logging as on the target machine(s).
        :param host: Host name or IP of the Server.
        :param port: 
        """
        self.objective_function = objective_function
        self.num_constraints = num_constraints
        self.username = username
        self.verbose = verbose
        try:
            self.host = socket.gethostbyname(host)
            self.port = int(port)
        except Exception as e:
            print("Invalid (host, port)!")
            print(e)
        self.max_gpu_usage = Value('d', 0.0)
        self.is_running = True
        self.lock_max_gpu_usage = Lock()
        
    def run(self):
        """Spawns 1 child Process, to evaluate objective function
        when receive a candidate. """        
        sv_conn = socket.socket()
        print('Waiting for connections from', self.host, ":", self.port)
        try:
            sv_conn.connect((self.host, self.port))
        except Exception as e:
            print("Can't connect!", str(e))
            sys.exit(0)
            
        sv_conn.setblocking(False) # Do not wait to recv()

        # Request the first candidate. Ready to work
        initial_message = {
            "observation": {},
            "gpu_info": get_all_gpu_processes_info()
        }
        self._send_dict_to_server(sv_conn, initial_message)
        
        # Evaluate Objective Function in a separate Process
        # Parent process sends candidates
        # Child process sends observation and print statements
        pconn, cconn = Pipe()
        objective_function_process = Process(target=self.evaluate_objective_function, \
                        args=(cconn,))
        objective_function_process.start()
        
        while self.is_running:
            # Send gpu info to the Server regularly
            gpu_info = get_all_gpu_processes_info()
            self._update_max_gpu_usage(gpu_info)
            sv_reply = {
                "gpu_info": get_all_gpu_processes_info()
            }
            self._send_dict_to_server(sv_conn, sv_reply)
            
            if self.verbose:
                self._send_dict_to_server(sv_conn, {"logging": "Handling"})
            try:
                # Handle response from Server
                sv_responses = sv_conn.recv(NUM_BYTES_RECEIVE).decode("utf8")
                for response in sv_responses.split('\n')[:-1]:
                    response = json.loads(response)
                    if "candidate" in response:
                        pconn.send(json.dumps(response["candidate"]))
                    if "command" in response:
                        pass # Feature to be developed
            except Exception as e:
                if "Resource temporarily unavailable" in str(e):
                    pass
                else:
                    print(e)
                    self.is_running = False
            
            # Check for messages from objective function process
            if self.verbose:
                self._send_dict_to_server(sv_conn, {"logging": "Checking"})
            if pconn.poll():
                obj_func_responses = pconn.recv()
                for response in obj_func_responses.split("\n")[:-1]:
                    response = json.loads(response)
                    if self.verbose:
                        self._send_dict_to_server(sv_conn, {"logging": "Objective func Response: " + str(response)})
                    
                    # Handle response from objective function process
                    # and relay message to the Server
                    sv_reply = {}
                    if "logging" in response:
                        sv_reply["logging"] = response["logging"]
                    if "objective" in response:
                        sv_reply["observation"] = response
                        with self.lock_max_gpu_usage:
                            max_gpu_usage = self.max_gpu_usage.value
                            self.max_gpu_usage.value = 0
                            self._send_dict_to_server(sv_conn, {"logging": "Max GPU usage: " + str(max_gpu_usage)})
                        if response["constraints"][0] > 0 and \
                                max_gpu_usage < MAXIMUM_ALLOWED_GPU_PERCENTAGE:
                            sv_reply["observation"]["contention_failure"] = True
                        else:
                            sv_reply["observation"]["contention_failure"] = False
                    if self.verbose:
                        self._send_dict_to_server(sv_conn, {"logging": "Server reply " + str(sv_reply)})
                    self._send_dict_to_server(sv_conn, sv_reply)
            
            # Interval of communication
            time.sleep(SERVER_TRAINER_MESSAGE_INTERVAL)
        p.kill()
        
    def _send_dict_to_server(self, sv_conn, d):
        try:
            sv_conn.sendall(str.encode(json.dumps(d) + "\n"))
        except Exception as e:
            print("Stopping...")
            print(e)
            self.is_running = False
            
    def _update_max_gpu_usage(self, gpu_info):
        total_gpu_mem = gpu_info["max_gpu"] 
        using_gpu_mem = 0
        for key in gpu_info:
            if key == "max_gpu":
                continue
            if gpu_info[key]["user"] == self.username:
                using_gpu_mem += gpu_info[key]["gpu_used"]
        gpu_usage = using_gpu_mem / total_gpu_mem
        if gpu_usage > self.max_gpu_usage.value:
            self.max_gpu_usage.value = gpu_usage
        
    def evaluate_objective_function(self, cconn):
        while True:            
            try:
                candidate = cconn.recv()
                candidate = json.loads(candidate)
            except Exception as e:
                print(e)
            # with redirect_print():
            
            start = datetime.now()
            try:
                print("Evaluating objective function")
                observation = self.objective_function(candidate)
                # Add GPU memory constraints
                with self.lock_max_gpu_usage:
                    observation['constraints'] = \
                            [self.max_gpu_usage.value - MAXIMUM_ALLOWED_GPU_PERCENTAGE] + \
                            observation["constraints"]
            except Exception as e:
                if 'out of memory' in str(e):
                    torch.cuda.empty_cache()
                    mean, variance = 0.001, 0.001
                    observation = {
                        "objective": [mean, variance],
                        "constraints": [1.1] + [0] * self.num_constraints
                    }
                else:
                    raise e
            elapsed = datetime.now() - start
                
            observation["time_started"] = start.strftime("%m/%d/%Y-%H:%M:%S")
            observation["time_elapsed"] = round(elapsed.seconds/3600, 2) # In hours, rounded to 2nd decimal
            
            # Add candidate into observation as well
            observation.update({"candidate": candidate})
            cconn.send(json.dumps(observation) + "\n")

