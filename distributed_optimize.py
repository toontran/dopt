import time
import sys
from multiprocessing import Process

import torch

from src.optimizers import NEIOptimizer
from src.distributed.submitMaster import processCommandsInParallel
from src.utils.config import CONFIG
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
        if torch.sum(X) - 3 <= 0:
            return True
        return False
    
    def outcome_constraint(X):
        # Is infeasible if > 0
        return torch.sum(X) - 3
    
    print("Starting optimizer..")
    optimizer = NEIOptimizer("hartmann.json", bounds, device="cpu",  is_possible=is_possible)
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
        
    # Bounds of the sample space
#     bounds = {
#         'batch_size': [2, 10],
#         'lr': [0.001, 10.0]
#     }
    bounds = {
        "x1": [0, 1],
        "x2": [0, 1],
        "x3": [0, 1],
        "x4": [0, 1],
        "x5": [0, 1],
        "x6": [0, 1],
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
    
    
    
