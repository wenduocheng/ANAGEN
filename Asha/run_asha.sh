#!/bin/bash

#SBATCH -p ma2-gpu
#SBATCH -w compute-4-13
#SBATCH --gres=gpu:1
#SBATCH --mem=23GB
#SBATCH --job-name asha-automation
#SBATCH --time=2-0:00:00
#SBATCH -o asha.out
#SBATCH -c 8
#SBATCH -e asha.err



python asha.py  > asha_2.log 2>&1
# python try_asha.py
