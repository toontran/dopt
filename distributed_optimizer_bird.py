import time
import os, sys
import argparse
from argparse import Namespace
from multiprocessing import Process
from typing import Dict, Any, Optional
import random

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
    "acet_update": ['tst008@acet116-lnx-10.bucknell.edu'], # To git pull
    "acet": [
        'tst008@acet116-lnx-11.bucknell.edu',
        'tst008@acet116-lnx-12.bucknell.edu',
#         'tst008@acet116-lnx-13.bucknell.edu',
#         'tst008@acet116-lnx-14.bucknell.edu',
#         'tst008@acet116-lnx-15.bucknell.edu',
#         'tst008@acet116-lnx-16.bucknell.edu',
#         'tst008@acet116-lnx-17.bucknell.edu',
#         'tst008@acet116-lnx-18.bucknell.edu',
#         'tst008@acet116-lnx-1.bucknell.edu',
#         'tst008@acet116-lnx-20.bucknell.edu',
#         'tst008@acet116-lnx-21.bucknell.edu',
    ],
#     "tung-torch": ['tung@jvs008-r1.bucknell.edu']
}
# Commands to run on target machines here
CONFIG["commands"] = {
    "acet_update": "cd PycharmProjects/distributed-optimizer/ && git pull" + \
                   " && module switch python/3.7-2020-05-28" + \
                   " && export LD_LIBRARY_PATH=/usr/remote/lib:/usr/remote/anaconda-3.7-2020-05-28/lib" + \
                   " && python3 ~/PycharmProjects/distributed-optimizer/distributed_optimizer_bird.py --run_as trainer",
    "acet": "sleep 10 && module switch python/3.7-2020-05-28" + \
                   " && export LD_LIBRARY_PATH=/usr/remote/lib:/usr/remote/anaconda-3.7-2020-05-28/lib" + \
                   " && python3 ~/PycharmProjects/distributed-optimizer/distributed_optimizer_bird.py --run_as trainer",
    "tung-torch": "/opt/anaconda/envs/jupyter37/bin/python ~/pj/dopt_v2/distributed_optimizer_bird.py --run_as trainer --data_folder ~/pj/dopt_v2/data/CroppedYale/"
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
    "username": "tst008",
    "num_constraints": 1
}
CONFIG["optimizer"] = {
    "bounds": {
        "x1": [0, 10],
        "x2": [0, 10]
    },
    "initial_candidates": [
        {"x1": 3, "x2": 6},
        {"x1": 4, "x2": 7}
    ],
    "device": "cpu",
    "seed": 0,
    "filename": "test.dopt"
}


def get_feasibility(candidate) -> float:
    x1, x2 = candidate.values()
    return -(x1 - x2 + 1.5)
    
def objective_function_torch_input(X):
    X = -X.view(-1, 2)
    mask_constraint = X[...,0] - X[...,1] + 1.5 > 0
    cos_x = torch.cos(X[...,0])
    sin_y = torch.sin(X[...,1])
    first_term = sin_y * torch.exp((1-cos_x)**2)
    second_term = cos_x * torch.exp((1-sin_y)**2)
    third_term = (X[...,0] - X[...,1])**2
    result = -(first_term + second_term + third_term) 
    Y = ((result + 120) / 230) #* mask_constraint 
    # Add noise
    se = torch.norm(X, dim=-1, keepdim=True) * 0.02
    Yvar = BASE_VAR + se * torch.rand_like(se)
    true_var = BASE_VAR + se
    Y = Y.view(-1, 1, 1) + torch.rand_like(se) * Yvar
    return Y, Yvar.view_as(Y)**2, true_var.view_as(Y)**2
    
# Plug in the objective function here
def objective_function(candidate):    
    feasibility = get_feasibility(candidate)
    if feasibility > 0:
        print("Infeasible!")
        observation = {
            "objective": [0.001, 0.001],
            "constraints": [feasibility]
        }
        return observation
    print("Returning the observation")
    
    # Simulate input
    X = torch.tensor([candidate["x1"], candidate["x2"]], dtype=float)
    Y, Yvar, _ = objective_function_torch_input(X)
    mean, variance = Y.item(), Yvar.item()
    
    time.sleep(random.randint(50, 90))
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
        seed=CONFIG["optimizer"]["seed"]
    )
    server = Server(optimizer, CONFIG, 
                    initial_candidates=CONFIG["optimizer"]["initial_candidates"])
    server.run()
    
def start_trainers():
    trainer = Trainer(
        objective_function, 
        CONFIG["trainer"]["username"],
        CONFIG["server"]["host"],
        CONFIG["server"]["port"],
        num_constraints=CONFIG["trainer"]["num_constraints"]
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
    
    
    
