import time
import sys

from multiprocessing import Process

from src.optimizers import NEIOptimizer
from src.distributed.submitMaster import processCommandsInParallel
from src.utils.config import CONFIG
# from test_trainer import NegHartmannTrainer

import warnings
warnings.filterwarnings("ignore")


print(CONFIG)


def start_optimizer(bounds):
    r"""Start the optimizer and listen to available trainers"""
    print("Starting optimizer..")
    optimizer = NEIOptimizer("hartmann.json", bounds, device="cpu")
    optimizer.run(host=None)

def start_trainers(num_trainers_active=2):
    r"""Connect to available machines and start trainers in parallel"""
    assert num_trainers_active <= len(CONFIG["distribute"]["computer_list"]), \
        "Numbers of trainers active at once cannot be greater than number of computers available!"
    commands = ["module switch python/3.7-2020-05-28" + \
                " && export LD_LIBRARY_PATH=/usr/remote/lib:/usr/remote/anaconda-3.7-2020-05-28/lib" + \
                " && python3 ~/PycharmProjects/summer/test_trainer.py"
                for i in range(num_trainers_active)]
#     commands = ["date" for i in range(num_trainers)]
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
    bounds = {
        'x1': (0,1),
        'x2': (0,1),
        'x3': (0,1),
        'x4': (0,1),
        'x5': (0,1),
        'x6': (0,1),
    }
    
    # Open two processes: One starts the optimizer, the
    # other starts all the trainers  
    # https://pymotw.com/2/multiprocessing/basics.html
    o = Process(target=start_optimizer, args=(bounds,))
    o.daemon = True
    
    t = Process(target=start_trainers, args=(3,))
    t.daemon = False

    o.start()
    t.start()
    
    while True:
        time.sleep(1)
        if not o.is_alive():
            break
    
    o.kill(), t.kill()

if __name__ == "__main__":
    main()
    
    
    
