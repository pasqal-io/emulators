#!/bin/bash
#SBATCH -c 16
#SBATCH -G 1

python3 benchmark.py
