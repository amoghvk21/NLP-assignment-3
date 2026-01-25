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

python3.12 -u -m main train-test nhmm upos --max-epochs 30 --save-path ./output/nhmm-upos-fast/save/nhmm-upos-fast.pt --res-path ./output/nhmm-upos-fast/results/nhmm-upos-fast.csv

echo "finished job"