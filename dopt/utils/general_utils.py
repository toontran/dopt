r"""
Helper functions
"""
import sys
import time
from contextlib import contextmanager
import torch
import random

@contextmanager
def add_prefix_to_print(prefix): 
    global is_new_line
    orig_write = sys.stdout.write
    is_new_line = True
    def new_write(*args, **kwargs):
        global is_new_line
        if args[0] == "\n":
            is_new_line = True
        elif is_new_line:
            orig_write("[" + str(prefix) + "]: ")
            is_new_line = False
        orig_write(*args, **kwargs)
    sys.stdout.write = new_write
    yield
    sys.stdout.write = orig_write

@contextmanager
def timer(label):
    import time
    start = time.time()
    yield
    print(f"[Process {label}] elasped in {time.time()-start}")
    
def get_output_shape(model, image_dim):
    return model(torch.rand(*(image_dim))).data.shape

def generate_seed():
    return random.randint(1, 100000)