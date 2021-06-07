import time
import os, sys
import json
from typing import Callable, Dict, Tuple, Union
from datetime import datetime
import traceback

import socket
from multiprocessing import Process, Pipe, Value
from threading import Lock

import logging
import logging.handlers
import pickle
import traceback

import torch

from dopt.utils import get_all_gpu_processes_info


NUM_BYTES_RECEIVE = 1024
MAXIMUM_ALLOWED_GPU_PERCENTAGE = 0.9
SERVER_TRAINER_MESSAGE_INTERVAL = 5


class PipeConnectionHandler(logging.Handler):
    """
    A handler class which writes logging records, to a pipe connection.
    """
    terminator = '\n'

    def __init__(self, conn):
        logging.Handler.__init__(self)
        self.conn = conn

    def emit(self, record):
        """
        Emit a record.
        """
        try:
            d = dict(record.__dict__)
            d["msg"] = record.getMessage()
            self.conn.send(json.dumps(d) + self.terminator)
        except:
            try:
                formatter = logging.Formatter(f'%(message)s')
                formatted_record = formatter.format(logging.makeLogRecord(dict(record.__dict__)))
                d = {"error": formatted_record}
                self.conn.send(json.dumps(d) + self.terminator)
            except:
                self.handleError(record)
            


class ModifiedSocketHandler(logging.handlers.SocketHandler):
    terminator = '\n'
    def emit(self, record):
        """
        Emit a record.

        Not pickling record, directly send json serialized dictionary
        of the record.
        """
        try:
            d = dict(record.__dict__)
            d["msg"] = record.getMessage()
            self.send(str.encode(json.dumps(d) + self.terminator, encoding="utf8"))
        except Exception:
            try:
                formatter = logging.Formatter(f'%(message)s')
                formatted_record = formatter.format(logging.makeLogRecord(dict(record.__dict__)))
                d = {"error": formatted_record}
                self.send(str.encode(json.dumps(d) + self.terminator, encoding="utf8"))
            except:
                self.handleError(record)


class Trainer:

    def __init__(self,
                 objective_function: Callable[[Dict, logging.Logger], Tuple],
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
            traceback.print_exc()
        self.max_gpu_usage = Value('d', 0.0)
        self.is_running = True

        self.logger = logging.getLogger('') # to log Led Observer output over a socket
        self.handler = ModifiedSocketHandler(host,port) # handler to write to socket
        self.logger.addHandler(self.handler)

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
            traceback.print_exc()
            sys.exit(0)

        sv_conn.setblocking(False) # Do not wait to recv()

        try:
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
                    "gpu_info": gpu_info
                }
                self._send_dict_to_server(sv_conn, sv_reply)

                self.logger.debug("Handling")
                try:
                    # Handle response from Server
                    sv_responses = sv_conn.recv(NUM_BYTES_RECEIVE).decode("utf8")
                    for response in sv_responses.split('\n')[:-1]:
                        response = json.loads(response)
                        if "candidate" in response:
                            pconn.send(json.dumps(response["candidate"])) # <---- This
                        if "command" in response:
                            pass # Feature to be developed
                except Exception as e:
                    if "Resource temporarily unavailable" in str(e):
                        pass
                    else:
                        raise

                # Check for messages from objective function process
                self.logger.debug("Checking")
                if pconn.poll():
                    obj_func_responses = pconn.recv()
                    for response in obj_func_responses.split("\n")[:-1]:
                        try:
                            response = json.loads(response)
                        except ValueError as e:
                            continue
                        self.logger.debug("Objective func Response: " + str(response))

                        # Handle response from objective function process
                        # and relay message to the Server
                        sv_reply = {}
                        if "objective" in response:
                            sv_reply["observation"] = response
                            with self.lock_max_gpu_usage:
                                max_gpu_usage = self.max_gpu_usage.value
                                self.max_gpu_usage.value = 0
                                self.logger.debug( "Max GPU usage: " + str(max_gpu_usage))
                            if response["constraints"][0] > 0 and \
                                    max_gpu_usage < MAXIMUM_ALLOWED_GPU_PERCENTAGE:
                                sv_reply["observation"]["contention_failure"] = True
                            else:
                                sv_reply["observation"]["contention_failure"] = False
                        if "error" in response:
                            self.logger.error(response["error"])
                        if "stack_info" in response:
                            # Log
                            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                            stringReceived = logging.makeLogRecord(response)
                            self.logger.handle(stringReceived)
                        self._send_dict_to_server(sv_conn, sv_reply)

                # Interval of communication
                time.sleep(SERVER_TRAINER_MESSAGE_INTERVAL)
            objective_function_process.kill()
        except:
            objective_function_process.kill()
            self.logger.exception("Error in Trainer.run()")
            

    def _send_dict_to_server(self, sv_conn, d):
        try:
            sv_conn.sendall(str.encode(json.dumps(d) + "\n", encoding="utf8"))
        except Exception as e:
            print("Stopping...")
            traceback.print_exc()
            self.is_running = False


    def _update_max_gpu_usage(self, gpu_info):
        total_gpu_mem = gpu_info["max_gpu"]
        using_gpu_mem = 0
        for key in gpu_info:
            if key == "max_gpu" or key == "time_updated":
                continue
            if gpu_info[key]["user"] == self.username:
                using_gpu_mem += gpu_info[key]["gpu_used"]
        gpu_usage = using_gpu_mem / total_gpu_mem
        if gpu_usage > self.max_gpu_usage.value:
            self.max_gpu_usage.value = gpu_usage
            

    def evaluate_objective_function(self, cconn):
        # Child logger will report to the main logger
        child_logger = logging.getLogger('child')
        conn_handler = PipeConnectionHandler(cconn)
#         conn_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        child_logger.addHandler(conn_handler)
        child_logger.setLevel(logging.DEBUG)
        try:
            while True:
                # Receive candidate
                candidate = cconn.recv()
                candidate = json.loads(candidate)

                # Train on candiate
                start = datetime.now()
                try:
                    child_logger.debug("Evaluating objective function")
                    observation = self.objective_function(candidate, child_logger)
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
                        child_logger.exception("Error in observing objective function")
                        
                elapsed = datetime.now() - start

                observation["time_started"] = start.strftime("%m/%d/%Y-%H:%M:%S")
                observation["time_elapsed"] = round(elapsed.seconds/3600, 2) # In hours, rounded to 2nd decimal

                # Add candidate into observation then send back to parent process
                observation.update({"candidate": candidate})
                cconn.send(json.dumps(observation) + "\n")
        except:
            child_logger.exception("Error in observing objective function")