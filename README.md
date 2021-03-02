# Recommender-System

A developing recommender system, implements in tensorflow 2.

Dataset: MovieLens-100k, MovieLens-1m, MovieLens-20m, lastfm, Book-Crossing, and some satori knowledge graph.

Algorithm: UserCF, ItemCF, LFM, SLIM, GMF, MLP, NeuMF, FM, DeepFM, MKR, RippleNet, KGCN and so on.

Evaluation: ctr's auc f1 and topk's precision recall.

## Requirements

* Python 3.8
* Tensorflow 2.4.1

## Run

[Download data files](https://github.com/SSSxCCC/Recommender-System/tree/datafile) and put 'ds' and 'kg' under 'Recommender_System/data' folder.

Open parent directory of current file as project in PyCharm, set up Python 3.8 interpreter and pip install tensorflow==2.4.1.

Go to Recommender_System/algorithm/xxx/main.py and run.

---

# Recommender-System推荐系统

这是一个正在开发的基于tensorflow2实现的推荐系统。

数据集：电影MovieLens-100k, MovieLens-1m, MovieLens-20m，音乐lastfm，书Book-Crossing，以及一些satori知识图谱。

算法：UserCF（基于用户的协同过滤）, ItemCF（基于物品的协同过滤）, LFM, SLIM, GMF, MLP, NeuMF, FM, DeepFM, MKR, RippleNet, KGCN等。

评估指标：点击率预测ctr的auc和f1，topk评估的准确率precision和召回率recall.

## 需求

* Python 3.8
* Tensorflow 2.4.1

## 运行

[下载数据文件](https://github.com/SSSxCCC/Recommender-System/tree/datafile)并将文件夹'ds'和'kg'放到'Recommender_System/data'目录下。

在PyCharm里面将此文件的父文件夹作为项目打开，设置好Python3.8的环境并使用pip安装tensorflow的2.4.1版本。

到Recommender_System/algorithm/xxx/main.py源码文件下并点击运行。