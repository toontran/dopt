import time
import os, sys
import argparse
from argparse import Namespace
from multiprocessing import Process
from typing import Dict, Any, Optional

import torch
from torch import nn
import numpy as np

import dopt
from dopt import NEIOptimizer, Trainer, processCommandsInParallel
from dopt.synthetic_trainers import NegHartmannTrainer
from dopt.utils import get_output_shape
from test_objective_function import run_train_net_kfold # The objective function

import warnings
warnings.filterwarnings("ignore")

# The configurations
CONFIG = {}
CONFIG["distribute"] = {
    "computer_list": {
        "acet": [
#             'tst008@acet116-lnx-10.bucknell.edu',
#             'tst008@acet116-lnx-11.bucknell.edu',
#             'tst008@acet116-lnx-12.bucknell.edu',
#             'tst008@acet116-lnx-13.bucknell.edu',
#             'tst008@acet116-lnx-14.bucknell.edu',
#             'tst008@acet116-lnx-15.bucknell.edu',
#             'tst008@acet116-lnx-16.bucknell.edu',
#             'tst008@acet116-lnx-17.bucknell.edu',
#             'tst008@acet116-lnx-18.bucknell.edu',
            'tst008@acet116-lnx-1.bucknell.edu',
#             'tst008@acet116-lnx-20.bucknell.edu',
#             'tst008@acet116-lnx-21.bucknell.edu',
        ],
        "localhost": ['tung@localhost']
    },
    "max_jobs": 1, # Num jobs per computer
    "min_gpu": 500, # TODO: Only use computers with GPU used lower than min_gpu (mbs)
}
CONFIG["optimizer"] = {
    "num_restarts": 10,  # ???
    "raw_samples": 500,  # Sample on GP using Sobel sequence
    "options": {
        "batch_limit": 5,
        "max_iter": 200,
        "seed": 0
    }
}
# Commands to run on target machines here
# CONFIG["commands"] = {
#          "acet": "module switch python/3.7-2020-05-28" + \
#                    " && export LD_LIBRARY_PATH=/usr/remote/lib:/usr/remote/anaconda-3.7-2020-05-28/lib" + \
#                    " && python3 ~/PycharmProjects/summer/distributed_optimizer.py --run_as client",
#     "localhost": "/opt/anaconda/envs/jupyter37/bin/python ~/summer/distributed_optimizer.py --run_as client --data_folder ~/summer/data/CroppedYale/"
# }
CONFIG["commands"] = {
    "acet": "sleep 20 && echo 'Hey'",
    "localhost": "sleep 20 && echo 'Hey'"
}


# Plug in the objective function here
class YaleFaceTrainer(Trainer):
    
    def __init__(self, 
                 args,
                 host: Optional[str] = "127.0.0.1",
                 port: Optional[str] = "15555"):
        super().__init__(host=host, port=port)
        self.args = args
        
    def get_feasibility(self, candidate) -> float:
        expected_input_shape = (1, 1, 192, 168)
        conv1_in = round(candidate["conv1"])
        conv1_kernel = round(candidate["conv1_kernel"])
        conv2_in = round(candidate["conv2"])
        conv2_kernel = round(candidate["conv2_kernel"])
        maxpool1_in = round(candidate["maxpool1"])
        maxpool2_in = round(candidate["maxpool2"])
        # Reconstruct the layers for calculation
        conv1 = nn.Conv2d(1, conv1_in, conv1_kernel, 1) 
        conv2 = nn.Conv2d(conv1_in, conv2_in, conv2_kernel, 1)
        maxpool1 = nn.MaxPool2d(maxpool1_in)
        maxpool2 = nn.MaxPool2d(maxpool2_in)
        
        try:
            conv1_out = get_output_shape(conv1, expected_input_shape)
            maxpool1_out = get_output_shape(maxpool1, conv1_out)
            conv2_out = get_output_shape(conv2, maxpool1_out)
            maxpool2_out = get_output_shape(maxpool2, conv2_out)
        except RuntimeError as e:
            if "Output size is too small" in str(e):
                return 2
            else:
                raise e
        fc1_in = np.prod(list(maxpool2_out)) # Flatten
        
        feasibility = -0.1 # Default is 0, meaning feasible, >0 means not feasible
        if conv1_out[-2] % maxpool1_in != 0 or \
            conv1_out[-1] % maxpool1_in != 0:
            feasibility += 1
        if conv2_out[-2] % maxpool2_in != 0 or \
            conv2_out[-1] % maxpool2_in != 0:
            feasibility += 1
        if conv1_in > conv2_in:
            feasibility += 1
        if fc1_in > 10**5:
            feasibility += 1
        print(f"Conv1: {get_output_shape(conv1, expected_input_shape)} % {maxpool1_in}, Conv2: {get_output_shape(conv2, maxpool1_out)} % {maxpool2_in}, Linear: {fc1_in}, Feasibility: {feasibility}")
        return feasibility
    
    def get_observation(self, candidate: Dict[str, Any]) \
            -> Dict[str, Any]:
        r""" Get observation by plugging the candidate into objective function.
        This method is made abstract to easier modify the objective function
        to run on different platforms.

        :param candidate:
        :return:
        """
        feasibility = self.get_feasibility(candidate)
        if feasibility > 0:
            print("Infeasible!")
            return 0.1, 0.1, feasibility
        
        # Simulate input
        input_args = Namespace(
            data_folder=self.args.data_folder,
            conv1=round(candidate["conv1"]),
            conv1_kernel=round(candidate["conv1_kernel"]),
            conv2=round(candidate["conv2"]),
            conv2_kernel=round(candidate["conv2_kernel"]),
            dropout1=candidate["dropout1"],
            maxpool1=round(candidate["maxpool1"]),
            maxpool2=round(candidate["maxpool2"]),
            no_cuda=False, 
            seed=1, 
            batch_size=round(candidate["batch_size"]),
            test_batch_size=1000,
            epochs=1,
            lr=candidate["lr"],
            gamma=0.7,
            log_interval=250, # was 250
            save_model=False,
            num_folds = 5
        )
        mean, variance = run_train_net_kfold(input_args)
        return mean, variance, feasibility


# Runs on the host machine
def start_optimizer():
    r"""Start the optimizer and listen to available trainers"""
    
#     def get_feasibility(X):
#         # Is infeasible if > 0
#         return torch.sum(X) - 2

    bounds = {
        "conv1": [3, 16],
        "conv1_kernel": [2, 10],
        "conv2": [20, 32],
        "conv2_kernel": [2, 10],
        "dropout1": [0.1, 1],
        "maxpool1": [2, 10],
        "maxpool2": [2, 10],
        'batch_size': [2, 7],
        'lr': [0.001, 10.0]
    }
#     bounds = {
#         "x1": [0,1],
#         "x2": [0,1],
#         "x3": [0,1],
#         "x4": [0,1],
#         "x5": [0,1],
#         "x6": [0,1],
#     }
    
    print("Starting optimizer..")
    # In case of huge infeasibility: If no previous observation or 
    # initial feasible candidate is found, random sampling is utilized,
    # making it hard to find a feasible sample. 
    # If already evaluated, skip it <-- 
#     initial_candidate = { 
#         "conv1": 3,
#         "conv1_kernel": 3,
#         "conv2": 20,
#         "conv2_kernel": 3,
#         "dropout1": 0.5,
#         "maxpool1": 2,
#         "maxpool2": 3,
#         'batch_size': 10,
#         'lr': 0.01
#     }
    initial_candidate = None
    
    optimizer = NEIOptimizer(
        "yaleface.json", bounds, 
        device="cpu",
        initial_candidate=initial_candidate,
        seed=1
    )
    optimizer.run(host=None, port=15556)

# Runs on the host machine
def start_trainers(num_trainers_active=2):
    r"""Connect to available machines and start trainers in parallel"""
#     assert num_trainers_active <= len(CONFIG["distribute"]["computer_list"]), \
#         "Numbers of trainers active at once cannot be greater than number of computers available!"

    commands = []
    for host_cat in CONFIG["distribute"]["computer_list"]:
        for host in CONFIG["distribute"]["computer_list"][host_cat]:
            commands.append({
                "category": host_cat, 
                "command": CONFIG["commands"][host_cat]
            })
    print("Starting trainers..")
    processCommandsInParallel(commands, CONFIG["distribute"])
    
# Calls start_optimizer and start_trainers simultaneously
def main():
    # Machines to be connected to
    print("Machine names:")
    for machine in CONFIG["distribute"]["computer_list"]:
        print(machine)
    
    # Open two processes: One starts the optimizer, the
    # other starts all the trainers  
    # https://pymotw.com/2/multiprocessing/basics.html
    o = Process(target=start_optimizer, args=())
    o.daemon = True
    t = Process(target=start_trainers, args=())
    t.daemon = False
    o.start()
    time.sleep(1) # Ensure the Optimizer starts before the Trainers
    t.start()
    while True:
        if not o.is_alive():
            print("Optimizer stopped. Terminating processes.")
            break
    
    o.kill(), t.kill()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='distributed_optimizer.py',
                                     description='''Optimize objective function of specified by a `Trainer`''')
    parser.add_argument('--run_as', action='store', dest='run_as',
                           help='Specify the role of the machine (host or client). Defaults to host',
                           type=str, required=False,
                           default="host")
    parser.add_argument('--data_folder', action='store', dest='data_folder',
                           help='Specify the directory to the data folder (for clients only)',
                           type=str, required=False,
                           default="~/PycharmProjects/summer/data/CroppedYale/")
    args = parser.parse_args()
    
    # Can modify these code to accomodate more options
    # E.g. Run different Trainers on same task
    if args.run_as == "host":
        main()
    elif args.run_as == "client":
        # Runs on clients
        trainer = YaleFaceTrainer(args,
                                  host="jvs008-r1.bucknell.edu",
                                  port="15555")
#         trainer = NegHartmannTrainer(host="jvs008-r1.bucknell.edu",
#                                       port="15555")
        trainer.run()
    
    
    
