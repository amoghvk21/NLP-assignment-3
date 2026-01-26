# Unsupervised Learning for Part-of-Speech Tagging

This project employs a subset of Penn Treebank dataset and evaluates
HMM and K-means on the PoS tagging problem.

## Introduction

This project implements the following models and algorithms:

- Hidden Markov Model (HMM) + Expectation-Maximization (EM) algorithms:
  - Standard EM (EM) (the classic soft EM algorithm)
  - Stochastic EM (sEM)
  - Viterbi-EM (hard-EM)
  - Maximal Likelihood Estimation (MLE) (supervised learning) - Already implemented

- Neural HMM

- K-means over BERT Embeddings

HMM models employ log scale parameters to avoid underflow.

## Getting started

To train and test HMM with EM for 10 epochs and validate every 5 epochs on UPOS tags:

```python
cd skeleton_code
python -m main train-test hmm-hardEM upos --max-epochs 50 1 --save-path ./output/hmm-hard-em-upos-rand/save/hmm-hard-em-upos-rand.pt --res-path ./output/hmm-hard-em-upos-rand/results/hmm-hard-em-upos-rand.csv --reset-method random
```

Use `--subset` argument to specify the maximum rows of data to be used.

Use `--initial-guesses` argument to specify a `.pt` hmm model to use as the starting point for training

To check more argument usage, run `python -m main --help`.

FYI: I will be training using staged training for each epoch so that:
- Each stage is saved
- I get metrics for the entire dataset for each epoch
  - So that I can draw graphs (each stage evals on whole dataset rather than 5%)

## Repository structure

```
.
|  AmoghVishwakarmaReport2526.pdf      # my report
|  requirements.txt                    # python requirements
|  experiments.sh                      # all experiments ran for the report
|  Neural HMM Paper (Tran et al.).pdf  # Paper for Neural HMM implementation
|  Online EM Paper.pdf                 # Paper for SEM implementation
└─skeleton_code
    │  argparser.py                    # argument parser
    │  hmm_pipeline.py                 # HMM training and testing pipelines
    │  kmeans_pipeline.py              # K-meand with BERT embeddings training and testing pipelines
    │  logging_nlp.py                  # logger setup
    │  main.py                         # main
    │  nhmm_pipeline.py                # Neural HMM training and testing pipelines
    │  preprocess_dataset.py           # dataset loading and preprocessing
    │  ptb-train.conllu                # Penn Treebank subset dataset
    │  README.md
    |  results_parser.ipynb            # notebook to generate graphs displayed in the report
    │  utils.py                        # auxiliary functions
    └─pos_tagging
        │  base.py
        │  hmm.py                      # HMM model
        │  kmeans.py                   # K-means model
        └─ nhmm.py                     # Neural HMM model
```
