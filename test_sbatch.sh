#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:01:00
#SBATCH --output=test_sbatch.out


cd $HOME/multinomial-dawid-skene/

module purge
module load 2022
module load Anaconda3/2022.05
module load Python/3.10.4-GCCcore-11.3.0-bare

srun python data_loading.py --datapoints 4000 --iterations 1 --algo ds