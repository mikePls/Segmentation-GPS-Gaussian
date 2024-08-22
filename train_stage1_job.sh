#!/bin/bash
#$ -cwd                     # Set the working directory for the job to the current directory
#$ -j y                     # Join standard out (stdout) and standard error (stderr) streams
#$ -pe smp 12                # Request 8 CPU cores
#$ -l h_rt=30:0:0           # Request runtime
#$ -l h_vmem=7.5G            # Request 11GB of memory per core
#$ -l gpu=1                 # Request 1 GPU
#$ -N GPS-Stage1                 # Name the job "debug"
#$ -o train_stage1.out      # Specify the output file
#$ -e train_stage1.err      # Specify the error file
#$ -m ea                   # Send email at the beginning, end, and in case of an abort
#$ -M michael_panagopoulos@hotmail.com  # Email address to send notifications

# Load Anaconda module
module load anaconda3

# Activate conda environment
source activate gps_gaussian

# Run the training script
python ct_stage1.py