#!/bin/bash
#SBATCH -p short # partition (queue) 
#SBATCH -N 1 # (leave at 1 unless using multi-node specific code) 
#SBATCH -n 1 # number of cores 
#SBATCH --mem-per-cpu=8192 # memory per core 
#SBATCH --job-name="myjob" # job name 
#SBATCH -o slurm.%N.%j.stdout.txt # STDOUT 
#SBATCH -e slurm.%N.%j.stderr.txt # STDERR 
#SBATCH --mail-user=username@bucknell.edu # address to email 
#SBATCH --mail-type=ALL # mail events (NONE, BEGIN, END, FAIL, ALL) 
for i in {1..100000}; do 
  echo $RANDOM >> SomeRandomNumbers.txt 
done 
sort -n SomeRandomNumbers.txt
