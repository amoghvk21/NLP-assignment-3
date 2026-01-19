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

python3.12 -u -m main train-test hmm-EM upos --max-epochs 100 1 --save-path ./output/hmm-soft-em-upos-dir/save/hmm-soft-em-upos-dir.pt --res-path ./output/hmm-soft-em-upos-dir/results/hmm-soft-em-upos-dir.csv

echo "finished job"