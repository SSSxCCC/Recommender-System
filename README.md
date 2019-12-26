# Recommender-System

A developing recommender system, implements in tensorflow 2.

Dataset: MovieLens-100k, MovieLens-1m, MovieLens-20m, lastfm, and some satori knowledge graph.

Algorithm: UserCF, ItemCF, LFM, GMF, MLP, NeuMF, FM, DeepFM, MKR, RippleNet, KGCN and so on.

Evaluation: ctr's auc f1 and topk's precision recall.

## Requirements

* Python 3.7
* Tensorflow 2.1.0rc0

## Run

[Download data files](https://github.com/SSSxCCC/Recommender-System/tree/datafile) and put 'ds' and 'kg' under 'Recommender_System/data' folder.

Open parent directory of current file as project in PyCharm, set up Python 3.7 interpreter and pip install tensorflow==2.1.0rc0.

Go to Recommender_System/algorithm/xxx/main.py and run.
