#!/bin/bash
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH --mem=48G

source cesga_venv/bin/activate
cd src/
python3 main.py
