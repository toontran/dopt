import time
import sys

from multiprocessing import Process

from src.optimizers import NEIOptimizer
from src.distributed.submitMaster import processCommandsInParallel
from src.utils.config import CONFIG
# from test_trainer import NegHartmannTrainer


print(CONFIG)


def start_optimizer(bounds):
    r"""Start the optimizer and listen to available trainers"""
    optimizer = NEIOptimizer(bounds, device="cpu")
    optimizer.run()

def start_trainers(num_trainers=2):
    r"""Connect to available machines and start trainers in parallel"""
    assert num_trainers <= len(CONFIG["distribute"]["computer_list"])
    commands = ["python3 ~/PycharmProjects/summer/test_trainer.py"]
    print("Starting trainers..")
    sys.stdout.flush()
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
        'x6': (0,1)
    }
    
    # Open two processes: One starts the optimizer, the
    # other starts all the trainers  
    # https://pymotw.com/2/multiprocessing/basics.html
    t = Process(target=start_trainers, args=(3,))
    t.daemon = False
    
    o = Process(target=start_optimizer, args=(bounds,))
    o.daemon = True
#     s = pool.apply_async(whileTrue, [100])
#     p = pool.apply_async(whileTrue, [1000])

#     pool.close()
#     pool.join()
# #     s.get(), p.get()
#     pattern.get(), parsed.get()
    t.start()
    time.sleep(1)
    o.start()


if __name__ == "__main__":
    main()
    
    
    
