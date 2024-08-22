#!/bin/bash
#$ -cwd
#$ -j y
#$ -N reproduce_nna
#$ -o ./exp_
#$ -e ./exp_
#$ -pe smp 8            # [8, 12] cores per GPU
#$ -l h_rt=4:0:0      # 240 hours runtime
#$ -l h_vmem=11G        # 11G RAM per core
#$ -l gpu=1
module load anaconda3
module load cuda/11.0
module load gcc/8.2

source activate gps_gaussian
python /data/home/ec23984/code/GPS-Gaussian/train_stage1.py            
