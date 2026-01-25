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

python3.12 -u -m main train-test nhmm upos --max-epochs 20 --save-path ./output/save/nhmm-upos/nhmm.pt --res-path ./output/results/nhmm-upos/nhmm_results.csv

echo "finished job"