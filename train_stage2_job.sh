#!/bin/bash
#$ -cwd                     # Set the working directory for the job to the current directory
#$ -j y                     # Join standard out (stdout) and standard error (stderr) streams
#$ -pe smp 12                # Request 8 CPU cores
#$ -l h_rt=48:0:0           # Request runtime
#$ -l h_vmem=7.5G            # Request 11GB of memory per core
#$ -l gpu=1                 # Request 1 GPU
#$ -N GPS-Stage2                 # Name the job 
#$ -o train_stage2.out      # Specify the output file
#$ -e train_stage2.err      # Specify the error file
#$ -m e                   # Send email at the beginning, end, and in case of an abort 'bea'
#$ -M michael_panagopoulos@hotmail.com  # Email address to send notifications

# Load Anaconda module
module load anaconda3

# Activate conda environment
source activate gps_gaussian

# Run the training script
python ct_stage2.py