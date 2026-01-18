#!/bin/bash


# --max-epochs 3 5 means 3 stages with 5 epochs each (for a total of 3x5 = 15 epochs)


# HMM MLE train and test upos   (hmm-mle-upos)
python -m main train-test hmm-mle upos --save-path ./save/hmm-mle-upos.pt --res-path ./results/hmm-mle-upos.csv

# HMM MLE train and test xpos   (hmm-mle-xpos)
python -m main train-test hmm-mle xpos --save-path ./save/hmm-mle-xpos.pt --res-path ./results/hmm-mle-xpos.csv

##############################################################

# HMM Soft EM train and test upos 100 epochs   (hmm-em-upos)
python -m main train-test hmm-EM upos --max-epochs 100 1 --save-path ./save/hmm-em-upos/hmm-em-upos.pt --res-path ./results/hmm-em-upos/hmm-em-upos.csv

# HMM Soft EM train and test xpos 100 epochs   (hmm-em-xpos)
python -m main train-test hmm-EM xpos --max-epochs 100 1 --save-path ./save/hmm-em-xpos/hmm-em-xpos.pt --res-path ./results/hmm-em-xpos/hmm-em-xpos.csv

##############################################################

# HMM Hard EM train and test upos 20 epochs   (hmm-hard-em-upos)
python -m main train-test hmm-hardEM upos --max-epochs 20 1 --save-path ./save/hmm-hard-em-upos/hmm-hard-em-upos.pt --res-path ./results/hmm-hard-em-upos/hmm-hard-em-upos.csv

# HMM Hard EM train and test xpos 20 epochs   (hmm-hard-em-xpos)
python -m main train-test hmm-hardEM xpos --max-epochs 20 1 --save-path ./save/hmm-hard-em-xpos/hmm-hard-em-xpos.pt --res-path ./results/hmm-hard-em-xpos/hmm-hard-em-xpos.csv

# HMM Hard EM train and test upos 20 epocs with initial guess from sEM (hmm-hard-em-upos_2)
python -m main train-test hmm-hardEM upos --max-epochs 20 1 --save-path ./save/hmm-hard-em-upos_2/hmm-hard-em-upos_2.pt --res-path ./results/hmm-hard-em-upos_2/hmm-hard-em-upos_2.csv --initial-guess ./save/hmm-sem-upos/hmm-sem.0.pt 

##############################################################

# HMM sEM train and test upos 20 epochs   (hmm-sem-upos)
python -m main train-test hmm-sEM upos --max-epochs 20 1 --save-path ./save/hmm-sem-upos/hmm-sem.pt --res-path ./results/hmm-sem-upos/hmm-sem.csv

# HMM sEM train and test xpos 20 epochs   (hmm-sem-xpos)
python -m main train-test hmm-sEM xpos --max-epochs 25 1 --save-path ./save/hmm-sem-xpos/hmm-sem.pt --res-path ./results/hmm-sem-xpos/hmm-sem.csv

##############################################################

# Neural HMM train and test upos 20 epochs   (nhmm-upos)
python -m main train-test nhmm upos --max-epochs 20 1 --save-path ./save/nhmm-upos/nhmm.pt --res-path ./results/nhmm-upos/nhmm.csv

# Neural HMM train and test xpos 20 epochs   (nhmm-xpos)
python -m main train-test nhmm xpos --max-epochs 20 1 --save-path ./save/nhmm-xpos/nhmm.pt --res-path ./results/nhmm-xpos/nhmm.csv

##############################################################

# K-means train and test upos 20 epochs   (kmeans-upos)
python -m main train-test kmeans upos --max-epochs 20 1 --save-path ./save/kmeans-upos/kmeans.pt --res-path ./results/kmeans-upos/kmeans.csv

# K-means train and test xpos 20 epochs   (kmeans-xpos)