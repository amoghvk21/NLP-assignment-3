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

python3.12 -u -m main train-test hmm-hardEM xpos --max-epochs 50 1 --save-path ./output/hmm-hard-em-xpos-dir/save/hmm-hard-em-xpos-dir.pt --res-path ./output/hmm-hard-em-xpos-dir/results/hmm-hard-em-xpos-dir.csv

echo "finished job"