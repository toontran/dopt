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
from dopt import NEIOptimizer, Trainer, Server
from dopt.utils import get_output_shape
from test_objective_function import run_train_net_kfold # The objective function

import warnings
warnings.filterwarnings("ignore")

# The configurations
CONFIG = {}
CONFIG["computer_list"] = {
#     "acet": [
#         'tst008@acet116-lnx-10.bucknell.edu',
#         'tst008@acet116-lnx-11.bucknell.edu',
#         'tst008@acet116-lnx-12.bucknell.edu',
#         'tst008@acet116-lnx-13.bucknell.edu',
#         'tst008@acet116-lnx-14.bucknell.edu',
#         'tst008@acet116-lnx-15.bucknell.edu',
#         'tst008@acet116-lnx-16.bucknell.edu',
#         'tst008@acet116-lnx-17.bucknell.edu',
#         'tst008@acet116-lnx-18.bucknell.edu',
#         'tst008@acet116-lnx-1.bucknell.edu',
#         'tst008@acet116-lnx-20.bucknell.edu',
#         'tst008@acet116-lnx-21.bucknell.edu',
#     ],
    "tung-torch": ['tung@jvs008-r1.bucknell.edu']
}
# Commands to run on target machines here
CONFIG["commands"] = {
         "acet": "module switch python/3.7-2020-05-28" + \
                   " && export LD_LIBRARY_PATH=/usr/remote/lib:/usr/remote/anaconda-3.7-2020-05-28/lib" + \
                   " && python3 ~/PycharmProjects/distributed-optimizer/distributed_optimizer.py --run_as trainer",
    "tung-torch": "/opt/anaconda/envs/jupyter37/bin/python ~/pj/dopt_v2/distributed_optimizer.py --run_as trainer --data_folder ~/pj/dopt_v2/data/CroppedYale/"
}
#
# CONFIG["commands"] = {
#     "acet": "sleep 20 && echo 'Hey'",
#     "tung-torch": "sleep 20 && echo 'Hey'"
# }
CONFIG["server"] = {
    "host": "jvs008-r1.bucknell.edu",
    "port": 15555
}
CONFIG["trainer"] = {
    "username": "tst008"
}
CONFIG["optimizer"] = {
    "bounds": {
        "conv1": [3, 16],
        "conv1_kernel": [2, 10],
        "conv2": [20, 32],
        "conv2_kernel": [2, 10],
        "dropout1": [0.1, 1],
        "maxpool1": [2, 10],
        "maxpool2": [2, 10],
        'batch_size': [2, 7],
        'lr': [0.001, 10.0]
    },
    "initial_candidates": None,
    "device": "cpu",
    "seed": 0,
    "filename": "yaleface.json"
}


def get_feasibility(candidate) -> float:
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
    if conv1_in > conv2_in:
        feasibility += 1
    if fc1_in > 10**5:
        feasibility += 1
    print(f"Conv1: {get_output_shape(conv1, expected_input_shape)} % {maxpool1_in}, Conv2: {get_output_shape(conv2, maxpool1_out)} % {maxpool2_in}, Linear: {fc1_in}, Feasibility: {feasibility}")
    return feasibility

# Plug in the objective function here
def objective_function(data_folder, candidate):
    feasibility = get_feasibility(candidate)
    if feasibility > 0:
        print("Infeasible!")
        observation = {
            "objective": [0.1, 0.1],
            "constraints": [feasibility]
        }
        return observation
    print("Returning the observation")
#     time.sleep(10)
#     observation = {
#             "objective": [0.1, 0.1],
#             "constraints": [feasibility]
#     }
#     print("Observation prepared. Sending..")
#     return observation
    # Simulate input
    input_args = Namespace(
        data_folder=data_folder,
        conv1=round(candidate["conv1"]),
        conv1_kernel=round(candidate["conv1_kernel"]),
        conv2=round(candidate["conv2"]),
        conv2_kernel=round(candidate["conv2_kernel"]),
        dropout1=candidate["dropout1"],
        maxpool1=round(candidate["maxpool1"]),
        maxpool2=round(candidate["maxpool2"]),
        no_cuda=False, 
        seed=2, 
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

    observation = {
        "objective": [mean, variance],
        "constraints": [feasibility]
    }
    return observation

    
# Calls start_optimizer and start_trainers simultaneously
def start_server():
    optimizer = NEIOptimizer(
        CONFIG["optimizer"]["filename"], 
        CONFIG["optimizer"]["bounds"], 
        device=CONFIG["optimizer"]["device"],
#         initial_candidate=CONFIG["initial_candidate"], PRIORITY <-------------
        seed=CONFIG["optimizer"]["seed"]
    )
    server = Server(optimizer, CONFIG, 
                    initial_candidates=CONFIG["optimizer"]["initial_candidates"])
    server.run()
    
def start_trainers():
    from functools import partial
    of = partial(objective_function, args.data_folder)
    trainer = Trainer(
        of, 
        CONFIG["trainer"]["username"],
        CONFIG["server"]["host"],
        CONFIG["server"]["port"]
    )
    trainer.run()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='distributed_optimizer.py',
                                     description='''Optimize objective function of specified by a `Trainer`''')
    parser.add_argument('--run_as', action='store', dest='run_as',
                           help='Specify the role of the machine (server or trainer). Defaults to server',
                           type=str, required=False,
                           default="server")
    parser.add_argument('--data_folder', action='store', dest='data_folder',
                           help='Specify the directory to the data folder (for clients only)',
                           type=str, required=False,
                           default="~/PycharmProjects/summer/data/CroppedYale/")
    args = parser.parse_args()
    
    # Can modify these code to accomodate more options
    # E.g. Run different Trainers on same task
    if args.run_as == "server":
        start_server()
    elif args.run_as == "trainer":
        start_trainers()
    
    
    
