import time
import sys
import subprocess


def check_host(host):
    """Checks if we can ssh to this host.
    
    :param host: Host to be ssh'd into
    :return:     None if host is invalid, else return host from input
    """
    try:
        if subprocess.check_call(['ssh', host, 'date'], \
                         stdout=subprocess.PIPE) == 0:
            return host
    except subprocess.CalledProcessError:
        print('Error: host %s not available.' % (host))

        
def process_commands_in_parallel(commands): # Name to be changed
    """Use "ssh [HOST] [COMMAND]" to run commands on remote machines.
    Only execute once for each machine. This function runs forever,
    i.e. never terminates unless the program is terminated.
    
    :param commands: A dictionary of {"host": [HOSTNAME], "command": [COMMANDS]}
    """
    
    for command in commands:
        host = check_host(command["host"])
        cmd = command["command"]
        
        if host == None:
            continue
        else:
            p = subprocess.Popen(['ssh', host, cmd])

            print('Submited to ' + host + ': ' + cmd)
        
    # Wait forever (Only escape using KeyboardInterrupt)
    while True:
        time.sleep(100)
        
        
if __name__ == "__main__":
    process_commands_in_parallel([
        {"host": "localhost", "command": "sleep 10 && echo BRUH"},
        {"host": "tst008@acet116-lnx-10.bucknell.edu", "command": "sleep 10 && echo COOL"}
    ])