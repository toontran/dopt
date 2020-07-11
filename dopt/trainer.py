import time
import os, sys
import json
from typing import Callable, Dict, Tuple, Union

import socket
from multiprocessing import Process, Pipe
from threading import Lock

from dopt.utils import get_all_gpu_processes_info


NUM_BYTES_RECEIVE = 1024
MAXIMUM_ALLOWED_GPU_PERCENTAGE = 0.9
SERVER_TRAINER_MESSAGE_INTERVAL = 5


class Trainer:
    
    def __init__(self, 
                 objective_function: Callable[[Dict], Tuple],
                 username: str,
                 host: str,
                 port: Union[int, str]):
        """
        
        :param username: The username we're logging as on the target machine(s)
        """
        self.objective_function = objective_function
        self.username = username
        try:
            self.host = socket.gethostbyname(host)
            self.port = int(port)
        except Exception as e:
            print("Invalid (host, port)!")
            print(e)
        self.max_gpu_usage = 0
        self.is_running = True
        self.a_lock = Lock()
        
    def run(self):
        """Spawns 1 child Process, to evaluate objective function
        when receive a candidate. """        
        trainer = socket.socket()
        print('Waiting for connections from', self.host, ":", self.port)
        try:
            trainer.connect((self.host, self.port))
        except Exception as e:
            print("Can't connect!", str(e))
            sys.exit(0)
            
        trainer.setblocking(False) # Do not wait to recv()

        # Request the first candidate. Ready to work
        initial_message = {
            "observation": {},
            "gpu_info": get_all_gpu_processes_info()
        }
        self._send_dict_to_server(trainer, initial_message)
        
        # Evaluate Objective Function in a separate Process
        # Parent process sends candidates
        # Child process sends observation and print statements
        pconn, cconn = Pipe()
        objective_function_process = Process(target=self.evaluate_objective_function, \
                        args=(cconn,trainer,))
        objective_function_process.start()
        
        while self.is_running:
            # Send gpu info to the Server regularly
            gpu_info = get_all_gpu_processes_info()
            self._update_max_gpu_usage(gpu_info)
            sv_reply = {
                "gpu_info": get_all_gpu_processes_info()
            }
            self._send_dict_to_server(trainer, sv_reply)
            
            self._send_dict_to_server(trainer, {"logging": "Handling"})
            try:
                # Handle response from Server
                sv_responses = trainer.recv(NUM_BYTES_RECEIVE).decode("utf8")
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
            self._send_dict_to_server(trainer, {"logging": "Checking"})
            if pconn.poll():
                obj_func_responses = pconn.recv()
                for response in obj_func_responses.split("\n")[:-1]:
                    response = json.loads(response)
                    self._send_dict_to_server(trainer, {"logging": "Objective func Response: " + str(response)})
                    
                    # Handle response from objective function process
                    # and relay message to the Server
                    sv_reply = {}
                    if "logging" in response:
                        sv_reply["logging"] = response["logging"]
                    if "objective" in response:
                        self._send_dict_to_server(trainer, {"logging": "Observation.. "})
                        sv_reply["observation"] = response
                        if response["constraints"][0] > 0 and \
                                self.max_gpu_usage < MAXIMUM_ALLOWED_GPU_PERCENTAGE:
                            sv_reply["observation"]["contention_failure"] = True
                        else:
                            sv_reply["observation"]["contention_failure"] = False
                    self._send_dict_to_server(trainer, {"logging": "Go"})
                    self._send_dict_to_server(trainer, {"logging": "Server reply " + str(sv_reply)})
                    self._send_dict_to_server(trainer, sv_reply)
            
            # Interval of communication
            # CAN'T SLEEP FOR SOME REASONS. WILL GET STUCK.
            time.sleep(SERVER_TRAINER_MESSAGE_INTERVAL)
        p.kill()
        
    def _send_dict_to_server(self, trainer, d):
        try:
            with self.a_lock:
                trainer.sendall(str.encode(json.dumps(d) + "\n"))
        except:
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
        if gpu_usage > self.max_gpu_usage:
            self.max_gpu_usage = gpu_usage
        
    def evaluate_objective_function(self, cconn, trainer):
        while True:            
            try:
                candidate = cconn.recv()
                candidate = json.loads(candidate)
            except Exception as e:
                print(e)
            # with redirect_print():
            try:
                print("Evaluating objective function")
                observation = self.objective_function(candidate)
            except Exception as e:
                print(e)
                print("ERROR")
            # Add candidate into observation as well
            observation.update({"candidate": candidate})
            self._send_dict_to_server(trainer, {"logging": "Obs: "+str(observation)})
            cconn.send(json.dumps(observation) + "\n")

