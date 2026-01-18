#! /bin/bash



echo "--------------------------------------------------------------------------------------------"
echo "Training HMM-HardEM XPOS"
python -m main train-test hmm-hardEM xpos --max-epochs 20 1 --save-path ./save/hmm-hard-em-xpos/hmm-hard-em-xpos.pt --res-path ./results/hmm-hard-em-xpos/hmm-hard-em-xpos.csv || true

echo "--------------------------------------------------------------------------------------------"
echo "Training HMM-HardEM XPOS with initial guesses"
python -m main train-test hmm-hardEM xpos --max-epochs 20 1 --save-path ./save/hmm-hard-em-xpos_2/hmm-hard-em-xpos_2.pt --res-path ./results/hmm-hard-em-xpos_2/hmm-hard-em-xpos_2.csv --initial-guesses ./save/hmm-sem-xpos/hmm-sem-xpos.0.pt || true

##############################################################

echo "--------------------------------------------------------------------------------------------"
echo "Training HMM-EM UPOS"
python -m main train-test hmm-EM upos --max-epochs 100 1 --save-path ./save/hmm-em-upos/hmm-em-upos.pt --res-path ./results/hmm-em-upos/hmm-em-upos.csv || true

echo "--------------------------------------------------------------------------------------------"
echo "Training HMM-EM XPOS"
python -m main train-test hmm-EM xpos --max-epochs 100 1 --save-path ./save/hmm-em-xpos/hmm-em-xpos.pt --res-path ./results/hmm-em-xpos/hmm-em-xpos.csv || true