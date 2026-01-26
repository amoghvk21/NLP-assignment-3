#!/bin/bash

echo "This job is running on: $(hostname)"

cd /home/av670/rds/hpc-work/NLP-assignment-3/skeleton_code

echo "Present working directory: $(pwd)"

echo "Job ID: $SLURM_JOB_ID"
echo "Job start time: $(date)"

module purge

echo "activating venv"
source ../venv/bin/activate

echo "running file"

python3.12 -u -m main train-test hmm-hardEM xpos --max-epochs 50 1 --save-path ./output/hmm-hard-em-xpos-rand/save/hmm-hard-em-xpos-rand.pt --res-path ./output/hmm-hard-em-xpos-rand/results/hmm-hard-em-xpos-rand.csv --reset-method random

echo "finished job"