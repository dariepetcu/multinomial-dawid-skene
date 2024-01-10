#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:01:00
#SBATCH --output=test_sbatch.out


cd $HOME/multinomial-dawid-skene/

conda activate scikit

srun python data_loading.py --datapoints 10000 --iterations 3 --algo mn