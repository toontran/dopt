r"""
Helper functions
"""
import time
from contextlib import contextmanager
import torch
import random


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