#!/bin/bash


# --max-epochs 3 5 means 3 stages with 5 epochs each (for a total of 3x5 = 15 epochs)

# HMM-MLE - 1 due to not staged 
python -m main train-test hmm-mle upos --max-epochs 1 5 --save-path ./save/hmm-mle.pt --res-path ./results/hmm-mle.csv

# HMM-HardEM (uses viterbi)
python -m main train-test hmm-hardEM upos --max-epochs 1 5 --save-path ./save/hmm-hard-em.pt --res-path ./results/hmm-hard-em.csv

# HMM-EM (soft EM)
python -m main train-test hmm-EM upos --max-epochs 1 5 --save-path ./save/hmm-em.pt --res-path ./results/hmm-em.csv

# HMM-sEM (sEM)
python -m main train-test hmm-sEM upos --max-epochs 1 5 --save-path ./save/hmm-sem.pt --res-path ./results/hmm-sem.csv

# K-Means
python -m main train-test kmeans xpos --word-embedding-path ./embeddings/bert_embeddings.pt --save-path ./save/kmeans_model.pkl --res-path ./results/kmeans_xpos_results.csv

# NHMM
python -m main train-test nhmm upos --max-epochs 1 5 --save-path ./save/nhmm_model.pt --res-path ./results/nhmm_upos_results.csv