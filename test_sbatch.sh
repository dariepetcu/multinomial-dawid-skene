#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:01:00
#SBATCH --output=test_sbatch.out


cd $HOME/multinomial-dawid-skene/

srun python data_loading.py --datapoints 4000 --iterations 1 --algo ds