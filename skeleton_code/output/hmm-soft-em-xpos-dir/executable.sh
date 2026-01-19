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

python3.12 -u -m main train-test hmm-EM xpos --max-epochs 100 1 --save-path ./output/hmm-soft-em-xpos-dir/save/hmm-soft-em-xpos-dir.pt --res-path ./output/hmm-soft-em-xpos-dir/results/hmm-soft-em-xpos-dir.csv

echo "finished job"