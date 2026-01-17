#!/bin/bash


# --max-epochs 3 5 means 3 stages with 5 epochs each (for a total of 3x5 = 15 epochs)


# HMM MLE train and test upos
python -m main train-test hmm-mle upos --save-path ./save/hmm-mle-upos.pt --res-path ./results/hmm-mle-upos.csv

# HMM MLE train and test xpos
python -m main train-test hmm-mle xpos --save-path ./save/hmm-mle-xpos.pt --res-path ./results/hmm-mle-xpos.csv



# HMM Soft EM train and test upos 100 epochs
python -m main train-test hmm-EM upos --max-epochs 100 1 --save-path ./save/hmm-em-upos/hmm-em-upos.pt --res-path ./results/hmm-em-upos/hmm-em-upos.csv

# HMM Soft EM train and test xpos 100 epochs
python -m main train-test hmm-EM xpos --max-epochs 100 1 --save-path ./save/hmm-em-xpos/hmm-em-xpos.pt --res-path ./results/hmm-em-xpos/hmm-em-xpos.csv



# HMM Hard EM train and test upos 20 epochs
python -m main train-test hmm-hardEM upos --max-epochs 20 1 --save-path ./save/hmm-hard-em-upos/hmm-hard-em-upos.pt --res-path ./results/hmm-hard-em-upos/hmm-hard-em-upos.csv

# HMM Hard EM train and test xpos 20 epochs
python -m main train-test hmm-hardEM xpos --max-epochs 20 1 --save-path ./save/hmm-hard-em-xpos/hmm-hard-em-xpos.pt --res-path ./results/hmm-hard-em-xpos/hmm-hard-em-xpos.csv



# HMM sEM train and test upos 20 epochs
python -m main train-test hmm-sEM upos --max-epochs 20 1 --save-path ./save/hmm-sem-upos/hmm-sem.pt --res-path ./results/hmm-sem-upos/hmm-sem.csv

# HMM sEM train and test xpos 20 epochs
python -m main train-test hmm-sEM xpos --max-epochs 20 1 --save-path ./save/hmm-sem-xpos/hmm-sem.pt --res-path ./results/hmm-sem-xpos/hmm-sem.csv