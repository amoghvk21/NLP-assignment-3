#!/bin/bash


# --max-epochs 3 5 means 3 stages with 5 epochs each (for a total of 3x5 = 15 epochs)


# HMM MLE train and test upos
python -m main train-test hmm-mle upos --save-path ./save/hmm-mle-upos.pt --res-path ./results/hmm-mle-upos.csv


# HMM MLE train and test xpos
python -m main train-test hmm-mle xpos --save-path ./save/hmm-mle-xpos.pt --res-path ./results/hmm-mle-xpos.csv



python -m main train-test hmm-EM upos --max-epochs 50 1 --save-path ./save/hmm-em-upos/hmm-em-upos.pt --res-path ./results/hmm-em-upos/hmm-em-upos.csv