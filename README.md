# CSC585 Algorithms for NLP 

# Assignment 3

* This repo is a simple runable instruction of [https://github.com/sheffieldnlp/fever-baselines]. More specifically, I choose to train the **Decomposable Attention model**. The MLP model also works, but due to the computing cost and time, I haven't trained that model. For the sampling method for NotEnoughINfo class, I tried both **Nearest-Page Sampling** and **Random Sampling**, you can choose either one of them to run.

## Full Credits
* Please refer to the [https://github.com/sheffieldnlp/fever-baselines] for more details.

## Pre-requisites
* Although the origin repo provides the conda installation instruction, I think using [Docker](https://www.docker.com/) is preferred, To enable GPU acceleration (run with `--runtime=nvidia`) once [NVIDIA Docker has been installed](https://github.com/NVIDIA/nvidia-docker). For this task, training with only CPU takes days, for my own run, I choose a computer with a Nvidia 1080Ti Graphic Card. (Thanks to Mingwei and his awesome computer)


## Docker Install

Download and run the latest FEVER. 
```
   $ docker volume create fever-data
   $ docker run -it -v fever-data:/fever/data sheffieldnlp/fever-baselines
```


## Download Data

### Wikipedia

To download a pre-processed Wikipedia dump ([license](https://s3-eu-west-1.amazonaws.com/fever.public/license.html)):
```
   $ bash scripts/download-processed-wiki.sh
```


### Dataset

Download the FEVER dataset from [their website](https://sheffieldnlp.github.io/fever/data.html) into the data directory:
```
   $ bash scripts/download-paper.sh
```
 
 
### Word Embeddings 
  
Download pretrained GloVe Vectors
```
   $ bash scripts/download-glove.sh
```


## Data Preparation
If you choose **Nearest-Page Sampling**
```
   $ PYTHONPATH=src python src/scripts/retrieval/document/batch_ir_ns.py --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 1 --split train
   $ PYTHONPATH=src python src/scripts/retrieval/document/batch_ir_ns.py --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 1 --split dev
```

If you choose **Random sampling**
```
   $ PYTHONPATH=src python src/scripts/dataset/neg_sample_evidence.py data/fever/fever.db
```


## Train DA
```
   #if using a CPU, set
   $ export CUDA_DEVICE=-1

   #if using a GPU, set
   $ export CUDA_DEVICE=0 #or cuda device id
```
Notice that the model has the same file name for both of two sampling methods, so you want to run both of them, make sure you change the file name of the model.

If you choose **Nearest-Page Sampling**
```
PYTHONPATH=src python src/scripts/rte/da/train_da.py data/fever/fever.db config/fever_nn_ora_sent.json logs/da_nn_sent --cuda-device $CUDA_DEVICE
mkdir -p data/models
cp logs/da_nn_sent/model.tar.gz data/models/decomposable_attention.tar.gz
```

If you choose **Random Sampling**
```
   # Using random sampled data for NotEnoughInfo (worse)
   $ PYTHONPATH=src python src/scripts/rte/da/train_da.py data/fever/fever.db config/fever_rs_ora_sent.json logs/da_rs_sent --cuda-device $CUDA_DEVICE
   $ mkdir -p data/models
   $ cp logs/da_rs_sent/model.tar.gz data/models/decomposable_attention.tar.gz
```

## Evaluation

These instructions are for the decomposable attention model.
 

### Evidence Retrieval Evaluation:

First retrieve the evidence for the dev/test sets:
```
   #Dev
   $ PYTHONPATH=src python src/scripts/retrieval/ir.py --db data/fever/fever.db --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file data/fever-data/dev.jsonl --out-file data/fever/dev.sentences.p5.s5.jsonl --max-page 5 --max-sent 5
    
   #Test
   $ PYTHONPATH=src python src/scripts/retrieval/ir.py --db data/fever/fever.db --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file data/fever-data/test.jsonl --out-file data/fever/test.sentences.p5.s5.jsonl --max-page 5 --max-sent 5

```
Then run the model:
```   
   #Dev
   $ PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db data/models/decomposable_attention.tar.gz data/fever/dev.sentences.p5.s5.jsonl  --log data/decomposable_attention.dev.log
    
   #Test
   $ PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db data/models/decomposable_attention.tar.gz data/fever/test.sentences.p5.s5.jsonl  --log data/decomposable_attention.test.log
```

## Scoring
### Score locally (for dev set)  
Score:
```
   $ PYTHONPATH=src python src/scripts/score.py --predicted_labels data/decomposable_attention.dev.log --predicted_evidence data/fever/dev.sentences.p5.s5.jsonl --actual data/fever-data/dev.jsonl
```
 
