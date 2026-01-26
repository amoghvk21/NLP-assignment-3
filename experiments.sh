#!/bin/bash


# ---------------------------------------------- #

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Need to be in this directory to run the below experiments
cd skeleton_code

#----------------------------------------------#

# HMM MLE UPOS
python -m main train-test hmm-mle upos --save-path ./output/hmm-mle-upos/save/hmm-mle-upos.pt --res-path ./output/hmm-mle-upos/results/hmm-mle-upos.csv 

# HMM MLE XPOS
python -m main train-test hmm-mle xpos --save-path ./output/hmm-mle-xpos/save/hmm-mle-xpos.pt --res-path ./output/hmm-mle-xpos/results/hmm-mle-xpos.csv

#----------------------------------------------#

# HMM Hard EM UPOS (dirichlet)
python -m main train-test hmm-hardEM upos --max-epochs 50 1 --save-path ./output/hmm-hard-em-upos-dir/save/hmm-hard-em-upos-dir.pt --res-path ./output/hmm-hard-em-upos-dir/results/hmm-hard-em-upos-dir.csv

# HMM Hard EM XPOS (dirichlet)
python -m main train-test hmm-hardEM xpos --max-epochs 50 1 --save-path ./output/hmm-hard-em-xpos-dir/save/hmm-hard-em-xpos-dir.pt --res-path ./output/hmm-hard-em-xpos-dir/results/hmm-hard-em-xpos-dir.csv

# HMM Hard EM UPOS (random)
python -m main train-test hmm-hardEM upos --max-epochs 50 1 --save-path ./output/hmm-hard-em-upos-rand/save/hmm-hard-em-upos-rand.pt --res-path ./output/hmm-hard-em-upos-rand/results/hmm-hard-em-upos-rand.csv --reset-method random

# HMM Hard EM XPOS (random)
python -m main train-test hmm-hardEM xpos --max-epochs 50 1 --save-path ./output/hmm-hard-em-xpos-rand/save/hmm-hard-em-xpos-rand.pt --res-path ./output/hmm-hard-em-xpos-rand/results/hmm-hard-em-xpos-rand.csv --reset-method random

#----------------------------------------------#

# HMM Soft EM UPOS (dirichlet)
python -m main train-test hmm-EM upos --max-epochs 100 1 --save-path ./output/hmm-soft-em-upos-dir/save/hmm-soft-em-upos-dir.pt --res-path ./output/hmm-soft-em-upos-dir/results/hmm-soft-em-upos-dir.csv

# HMM Soft EM XPOS (dirichlet)
python -m main train-test hmm-EM xpos --max-epochs 100 1 --save-path ./output/hmm-soft-em-xpos-dir/save/hmm-soft-em-xpos-dir.pt --res-path ./output/hmm-soft-em-xpos-dir/results/hmm-soft-em-xpos-dir.csv

#----------------------------------------------#

# HMM sEM UPOS alpha 0.6
python -m main train-test hmm-sEM upos --max-epochs 20 1 --save-path ./output/hmm-sem-upos-dir-alpha-0.6/save/hmm-sem-upos-dir-alpha-0.6.pt --res-path ./output/hmm-sem-upos-dir-alpha-0.6/results/hmm-sem-upos-dir-alpha-0.6.csv --alpha-sem 0.6

# HMM sEM UPOS alpha 0.7
python -m main train-test hmm-sEM upos --max-epochs 20 1 --save-path ./output/hmm-sem-upos-dir-alpha-0.7/save/hmm-sem-upos-dir-alpha-0.7.pt --res-path ./output/hmm-sem-upos-dir-alpha-0.7/results/hmm-sem-upos-dir-alpha-0.7.csv --alpha-sem 0.7

# HMM sEM UPOS alpha 0.8
python -m main train-test hmm-sEM upos --max-epochs 20 1 --save-path ./output/hmm-sem-upos-dir-alpha-0.8/save/hmm-sem-upos-dir-alpha-0.8.pt --res-path ./output/hmm-sem-upos-dir-alpha-0.8/results/hmm-sem-upos-dir-alpha-0.8.csv --alpha-sem 0.8

# HMM sEM UPOS alpha 0.9
python -m main train-test hmm-sEM upos --max-epochs 20 1 --save-path ./output/hmm-sem-upos-dir-alpha-0.9/save/hmm-sem-upos-dir-alpha-0.9.pt --res-path ./output/hmm-sem-upos-dir-alpha-0.9/results/hmm-sem-upos-dir-alpha-0.9.csv --alpha-sem 0.9

# HMM sEM UPOS alpha 1.0
python -m main train-test hmm-sEM upos --max-epochs 20 1 --save-path ./output/hmm-sem-upos-dir-alpha-1.0/save/hmm-sem-upos-dir-alpha-1.0.pt --res-path ./output/hmm-sem-upos-dir-alpha-1.0/results/hmm-sem-upos-dir-alpha-1.0.csv --alpha-sem 1.0

#----------------------------------------------#

# HMM sEM XPOS alpha 0.6
python -m main train-test hmm-sEM xpos --max-epochs 20 1 --save-path ./output/hmm-sem-xpos-dir-alpha-0.6/save/hmm-sem-xpos-dir-alpha-0.6.pt --res-path ./output/hmm-sem-xpos-dir-alpha-0.6/results/hmm-sem-xpos-dir-alpha-0.6.csv --alpha-sem 0.6

# HMM sEM XPOS alpha 0.7
python -m main train-test hmm-sEM xpos --max-epochs 20 1 --save-path ./output/hmm-sem-xpos-dir-alpha-0.7/save/hmm-sem-xpos-dir-alpha-0.7.pt --res-path ./output/hmm-sem-xpos-dir-alpha-0.7/results/hmm-sem-xpos-dir-alpha-0.7.csv --alpha-sem 0.7

# HMM sEM XPOS alpha 0.8
python -m main train-test hmm-sEM xpos --max-epochs 20 1 --save-path ./output/hmm-sem-xpos-dir-alpha-0.8/save/hmm-sem-xpos-dir-alpha-0.8.pt --res-path ./output/hmm-sem-xpos-dir-alpha-0.8/results/hmm-sem-xpos-dir-alpha-0.8.csv --alpha-sem 0.8

# HMM sEM XPOS alpha 0.9
python -m main train-test hmm-sEM xpos --max-epochs 20 1 --save-path ./output/hmm-sem-xpos-dir-alpha-0.9/save/hmm-sem-xpos-dir-alpha-0.9.pt --res-path ./output/hmm-sem-xpos-dir-alpha-0.9/results/hmm-sem-xpos-dir-alpha-0.9.csv --alpha-sem 0.9

# HMM sEM XPOS alpha 1.0
python -m main train-test hmm-sEM xpos --max-epochs 20 1 --save-path ./output/hmm-sem-xpos-dir-alpha-1.0/save/hmm-sem-xpos-dir-alpha-1.0.pt --res-path ./output/hmm-sem-xpos-dir-alpha-1.0/results/hmm-sem-xpos-dir-alpha-1.0.csv --alpha-sem 1.0

#----------------------------------------------#

# K-Means UPOS
python -m main train-test kmeans upos --word-embedding-path ./output/kmeans-upos/embeddings/bert_embeddings.pt --save-path ./output/kmeans-upos/save/kmeans-upos.pt --res-path ./output/kmeans-upos/results/kmeans-upos.csv

# K-Means XPOS
python -m main train-test kmeans xpos --word-embedding-path ./output/kmeans-xpos/embeddings/bert_embeddings.pt --save-path ./output/kmeans-xpos/save/kmeans-xpos.pt --res-path ./output/kmeans-xpos/results/kmeans-xpos.csv

#----------------------------------------------#

# NHMM UPOS
python -m main train-test nhmm upos --max-epochs 40 --save-path ./output/nhmm-upos/save/nhmm-upos.pt --res-path ./output/nhmm-upos/results/nhmm-upos.csv
# NHMM XPOS
python -m main train-test nhmm xpos --max-epochs 40 --save-path ./output/nhmm-xpos/save/nhmm-xpos.pt --res-path ./output/nhmm-xpos/results/nhmm-xpos.csv

#----------------------------------------------#