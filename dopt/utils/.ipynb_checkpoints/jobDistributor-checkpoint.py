"""
Joshua Stough
W&L, Image Group
July 2011
A job distributor class, that maintains a list of available machines,
and distributes jobs to them. My idea is that someone uses this by
generating their own command lines, like 'echo hello; cd ~/teaching; ls'
and this class basically makes the system call
'ssh host command'.  It's the caller's responsibility to generate
appropriate command lines, redirecting output as they see fit.

uses subprocess to do this, instantiating a process to execute the
ssh command.  Process information is then stored in a Job object,
which can in turn be queried.

This class should be instantiated by an independent thread, which
periodically checks for the completion of jobs and maintains a job
queue.  See submitMaster and testSubmitMaster for guidance. When
presented with a job to distribute when all hosts are busy to their
capacity, this code will hang, waiting for a spot to open up.  It's
someone else's responsibility to keep the original caller informed
about the submissions made--again, see submitMaster.

 License: This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3 of the License, or (at your
 option) any later version. This program is distributed in the hope that it
 will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 Public License for more details.
"""

import os, sys, shlex, subprocess, time
import json

sys.path.append(".")
from .listQueue import listQueue
# from dopt import CONFIG


# DIST_CONF = CONFIG["distribute"]


class Job(object):
    def __init__(self, host, proc, cmd_line):
        self.host = host
        self.proc = proc
        self.cmd_line = cmd_line
        self.startTime = time.ctime()

    def __str__(self):
        ret = '[Job on host %s: %-15s started %s]' % \
              (self.host, self.cmd_line[:15], self.startTime)
        return ret

    def poll(self):
        #return None for testing
        return self.proc.poll()

class JobDistributor(object):
    #Some static members.  Replace the elements of 
    #computer_list with hostnames you have ssh access to
    #without a password (see ssh-keygen)
#     computer_list = DIST_CONF["computer_list"]

#     maxJobs = DIST_CONF["max_jobs"]
    processes = {}  
    #dictionary associating hostname to a list of Job objects.
    totalJobs = 0
    instances = 0
    
    def __init__(self, computer_list, maxJobs):
        if self.instances == 1:
            raise AssertionError('JobDistributor init ERROR: ' + \
                                 'Only one JD object allowed')
        self.computer_list = computer_list
        self.maxJobs = maxJobs
        self.instances = 1
        for host_cat in self.computer_list:
            self.processes[host_cat] = {host:[] for host in self.computer_list[host_cat]}
        self.cleanComputerList()      

    def setMaxJobs(self, num):
        self.maxJobs = num

    def isFull(self):
        num_processes = sum([sum([1 for g in i.values()]) for i in self.processes.values()])
        return len(self) == num_processes*self.maxJobs

    def info(self):
        num_processes = sum([sum([1 for g in i.values()]) for i in self.processes.values()])
        return (num_processes, self.maxJobs*num_processes)

    def distribute(self, command):
        procNum = self.totalJobs
        #Simplified for now.  Just to see if the data all works.
        print('\nSearching for host for process %i...' % (procNum))
        hostFound = False
        waitCycles = 0
        command = json.loads(command)
        while not hostFound:
            try:
                host = self.getHost(command["category"])
                print('Host %s chosen for proc %i.' % (host, procNum))
                hostFound = True
                break
            except ValueError:
                print('%sWaiting to find host for proc %i.' % \
                      ('.'*waitCycles, procNum))
                waitCycles += 1
                time.sleep(waitCycles)
                #The exception here should not happen, because the
                #submitMaster that calls this ensures there is
                #availability.  This will hang if it was mistaken.
        #print('Starting command.')
        command_line = 'ssh ' + host + ' ' + command["command"]

        #This implies that shlex is maybe splitting up the command too much
        #in the case of matlab -nodisplay -r ... it thinks the commands
        #I want executed by matlab are for unix and I get syntax errors.
        #print('\n\nHere is the whole thing\n\n%s\n\nThere it was' % \
        #      (shlex.split(command_line)))

        #We'll build our own, for simplicity's sake.  That means
        #it is solely the responsibility of the caller to construct
        #the line as it should be run from the command line of the host.
        p = subprocess.Popen(['ssh', host, command["command"]])
        
        #p = subprocess.Popen(shlex.split(command_line))
        print('Submited to ' + host + ': ' + command["command"])
        self.processes[command["category"]][host].append(Job(host, p, command["command"]))
        self.totalJobs += 1

    def getHost(self, host_cat):
        """Find a host among the computer_list whose load is less than maxJobs."""
        #Could loop through computer_list here, but computer_list still lists the
        #unavailable ones.  
        for host in self.processes[host_cat]:
            
            #clean out finished jobs. Keep only those which haven't terminated.
            self.processes[host_cat][host] = [p for p in self.processes[host_cat][host]
                                              if p.poll() is None]
            
            if len(self.processes[host_cat][host]) < self.maxJobs:
                try:
                    if subprocess.check_call(shlex.split('ssh ' + host + ' date'), \
                                     stdout=subprocess.PIPE) == 0:
                        return host
                except subprocess.CalledProcessError:
                    print('getHost() error: host %s not available.' % (host))
                    
        #print('getHost(): could not find host.\n')
        raise ValueError('getHost() failed: could not find host.')

    def __str__(self):
        return '[JobDistributor: %i jobs on %i hosts]' % \
               (len(self), sum([len(self.processes[host_cat])
                               for host_cat in self.processes.values()]))
    
    def __len__(self):
        #Return number of jobs running
        return self.cleanup()

    def cleanup(self):
        num_jobs = 0
        for host_cat in self.processes:
            for host in self.processes[host_cat]:
                self.processes[host_cat][host] = [j for j in self.processes[host_cat][host] 
                                        if j.poll() is None]
                num_jobs += len(self.processes[host_cat][host])
        return num_jobs

    def cleanComputerList(self):
        for host_cat in self.computer_list:
            for host in self.computer_list[host_cat]:
                try:
                    subprocess.check_call(shlex.split('ssh ' + host + ' date'), \
                                         stdout=subprocess.PIPE);
                except Exception as e:
                    print(e)
                    print('cleanComputerList: host %s deemed unavailable, ignoring...' \
                          % (host))
                    self.processes[host_cat].pop(host)
                    
    





    
    
