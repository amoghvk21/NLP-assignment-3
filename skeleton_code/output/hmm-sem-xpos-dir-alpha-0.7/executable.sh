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

python3.12 -u -m main train-test hmm-sEM xpos --max-epochs 20 1 --save-path ./output/hmm-sem-xpos-dir-alpha-0.7/save/hmm-sem-xpos-dir-alpha-0.7.pt --res-path ./output/hmm-sem-xpos-dir-alpha-0.7/results/hmm-sem-xpos-dir-alpha-0.7.csv --alpha-sem 0.7

echo "finished job"