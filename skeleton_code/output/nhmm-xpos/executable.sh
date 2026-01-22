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

python3.12 -u -m main train-test nhmm xpos --max-epochs 20 1 --save-path ./output/nhmm-xpos/nhmm.pt --res-path ./output/nhmm-xpos/nhmm.csv

echo "finished job"