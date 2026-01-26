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

python3.12 -u -m main train-test nhmm upos --max-epochs 40 --save-path ./output/nhmm-upos-40/save_2/nhmm-upos.pt --res-path ./output/nhmm-upos-40/results_2/nhmm-upos-results.csv

echo "finished job"