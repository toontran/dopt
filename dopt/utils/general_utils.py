r"""
Helper functions
"""
import sys
import time
from contextlib import contextmanager
import torch
import random
import subprocess

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

def get_gpu_info():
    sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf8").split('\n')
    out_dict = {}
    for item in out_list:
        try:
            key, val = item.split(':')
            key, val = key.strip(), val.strip()
            if key in out_dict: # Already exists
                out_dict[key].append(val)
            else:
                out_dict[key] = [val]
        except:
            pass
    return out_dict

def get_general_info(pid):
    sp = subprocess.Popen(["ps", "-up", str(pid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out_str = sp.communicate()
    outputs = out_str[0].decode("utf8").split("\n")
    labels = outputs[0].split()
    info = outputs[1].split()
    if len(info) > len(labels): # Join commands that were splitted
        last_label_idx = len(labels)-1 
        info[last_label_idx] = " ".join(info[last_label_idx:])
        info = info[:len(labels)]
    process_info = {labels[i]: info[i] for i in range(len(info))}
    return process_info

def get_all_gpu_processes_info():
    processes = {}
    out_dict = get_gpu_info()
    max_gpu = int(out_dict["Total"][0].split()[0])
    
    processes["max_gpu"] = max_gpu
    for i, process_id in enumerate(out_dict["Process ID"]):
        process_info = get_general_info(process_id)
        processes[process_id] = {
            "name": out_dict["Name"][i],
            "user": process_info["USER"],
            "gpu_used": int(out_dict["Used GPU Memory"][i].split()[0]),
            "%cpu_used": float(process_info["%CPU"]),
            "%mem_used": float(process_info["%MEM"]),
            "command": process_info["COMMAND"]
        }
    return processes
    
