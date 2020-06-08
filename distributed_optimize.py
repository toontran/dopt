import time
import sys
from multiprocessing import Process

import torch
from torch import nn

from src.optimizers import NEIOptimizer
from src.distributed.submitMaster import processCommandsInParallel
from src.utils.config import CONFIG
from src.utils.torch_utils import get_output_shape
# from test_trainer import NegHartmannTrainer

import warnings
warnings.filterwarnings("ignore")


# Command to run on target machines
COMMAND = "module switch python/3.7-2020-05-28" + \
          " && export LD_LIBRARY_PATH=/usr/remote/lib:/usr/remote/anaconda-3.7-2020-05-28/lib" + \
          " && python3 ~/PycharmProjects/summer/run_trainer.py"
print("Using config: ", CONFIG)

def start_optimizer(bounds, is_feasible):
    r"""Start the optimizer and listen to available trainers"""
    
    def is_possible(X):
        expected_input_shape = (1, 1, 192, 168)
        conv1_in = round(float(X[0]))
        conv1_kernel = round(float(X[1]))
        conv2_in = round(float(X[2]))
        conv2_kernel = round(float(X[3]))
        maxpool1_in = round(float(X[5]))
        maxpool2_in = round(float(X[6]))
        # Reconstruct the layers for calculation
        conv1 = nn.Conv2d(1, conv1_in, conv1_kernel, 1) 
        conv2 = nn.Conv2d(conv1_in, conv2_in, conv2_kernel, 1)
        maxpool1 = nn.MaxPool2d(maxpool1_in)
        maxpool2 = nn.MaxPool2d(maxpool2_in)
        conv1_out = get_output_shape(
            maxpool1, get_output_shape(conv1, expected_input_shape)
        )
        conv2_out = get_output_shape(
            maxpool2, get_output_shape(conv2, conv1_out)
        ) 
        
        if conv1_out[-2] % maxpool1_in == 0 and \
            conv1_out[-1] % maxpool1_in == 0:
            return True
        if conv2_out[-2] % maxpool2_in == 0 and \
            conv2_out[-1] % maxpool2_in == 0:
            return True
        if conv1_in < conv2_in:
            return True
        return False
    
#     def outcome_constraint(X):
#         # Is infeasible if > 0
#         return torch.sum(X) - 3
    
    print("Starting optimizer..")
    optimizer = NEIOptimizer("yaleface.json", bounds, device="cpu",  is_possible=is_possible)
    optimizer.run(host=None)

def start_trainers(num_trainers_active=2):
    r"""Connect to available machines and start trainers in parallel"""
    assert num_trainers_active <= len(CONFIG["distribute"]["computer_list"]), \
        "Numbers of trainers active at once cannot be greater than number of computers available!"
    commands = [COMMAND for i in range(num_trainers_active)]
    print("Starting trainers..")
    processCommandsInParallel(commands)
    
def whileTrue(num_sec):
    start = time.time()
    while start-time.time() < num_sec:
        sleep(1)
        print("num_sec", end="")

def main():
    # Machines to be connected to
    print("Machine names:")
    for machine in CONFIG["distribute"]["computer_list"]:
        print(machine)
        
    # TODO: Deal with ordering problem
    bounds = {
        "conv1": [16, 128],
        "conv1_stride": [2, 10],
        "conv2": [64, 256],
        "conv2_stride": [2, 10],
        "dropout1": [0, 1],
        "maxpool1": [0, 10],
        "maxpool2": [0, 10],
        'batch_size': [2, 10],
        'lr': [0.001, 10.0]
    }
    
    # Open two processes: One starts the optimizer, the
    # other starts all the trainers  
    # https://pymotw.com/2/multiprocessing/basics.html
    o = Process(target=start_optimizer, args=(bounds, lambda x: True))
    o.daemon = True
    
    t = Process(target=start_trainers, args=(1,))
    t.daemon = False

    o.start()
    time.sleep(1)
    t.start()
    
    while True:
        time.sleep(1)
        if not o.is_alive():
            break
    
    o.kill(), t.kill()

if __name__ == "__main__":
    main()
    
    
    
