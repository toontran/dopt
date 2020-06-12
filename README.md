### How to use

    We'll be running `distributed_optimizer.py` to start the optimization.
    
    Essential Parts of the Script:
    - A MyWhateverTrainer that inherits the `Trainer` abstract class from 
    `src.trainer`, and implement the abstract method `get_observation`, in which
    the set of hyperparameters (candidate) given will be plugged into the 
    objective function.
    
    - A `main()` function that will be used to call 2 processes: one that
    start the `Optimizer` and one that runs appropriate sequence of commands to start
    `Trainers` on respective machines.
    
    - A `parser` that parse command line input. This is to switch the 
    `distributed_optimizer.py`file between 2 modes: host and client; and to 
    specify the path to the data file (it might be different on different machines).
    
    - adsd

    - Step 1: Make sure you have a copy of the distributed_optimizer.py and 
    related files on all of the machines you intend to use
    
    - Step 2: 