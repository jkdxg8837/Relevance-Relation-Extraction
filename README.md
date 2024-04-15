## Relevance-Relation-Extraction
This is an implementation code for "RRE: A relevance relation extraction framework for cross-domain recommender system at Alipay".
In this paper, we study the relationship between the head and tail entities in the knowledge graph. As a result, we propose a new downstream task called Relation Relevance Extraction(RRE) accompanied by a human-annotated dataset in a real-world Alipay scenario. What's more, we use this score in the downstream recommendation task to prove it can mitigate popularity bias in item-centric and user-centric tasks.
We use SimKGC as our baseline. Despite the topology generating and fusing module, the rest of our model follows a plug-in style, which indicates you can construct dataset and run the code by following the instruction provided by SimKGC.

## Requirements
* python>=3.7
* torch>=1.6 (for mixed precision training)
* transformers>=4.15

All experiments are run with 1 V100(32GB) GPUs.

## How to Run

It involves 3 steps: dataset preprocessing, model training, and model evaluation.


For WN18RR and FB15k237 datasets, we use files from [KG-BERT](https://github.com/yao8839836/kg-bert) and [SimKGC/data/WN18RR](https://github.com/intfloat/SimKGC/tree/main/data/WN18RR)

### WN18RR dataset

Step 1, preprocess the dataset
```
bash scripts/preprocess.sh WN18RR
```


Step 2, generate topology info

Unlike the open source dataset, our mini-program & content dataset has clearly defined the level of strength of relevance. So in the topology generation of our dataset, we connect edges except for "no correlation". As for open source datasets, take WN18RR as an example, we measure the semantic meaning of the corresponding 13 relation types, then decides the directions of the edge.
```
pip install networkx
pip install node2vec
bash scripts/generate_topo.sh
```
By replacing the right path of your train.txt and test.txt of WN18RR, you can get an embedding file under output path.

Step 3, training the model and (optionally) specify the output directory (< 3 hours), and replace the g_node2vec.emb path with the right path corresponding to your current dataset.
```
OUTPUT_DIR=./checkpoint/wn18rr/ bash scripts/train_wn.sh
```

Step 4, evaluate a trained model
```
bash scripts/eval.sh ./checkpoint/wn18rr/model_last.mdl WN18RR
```


Also, there exists some triggers to control modules used in our model, located in models_no_hr_vector.py. If you don't want to use momentum learning or topology info, feel free to set FALSE to them.


### Acknowledgements
Thanks Wang et.al for their open source code [SimKGC](https://github.com/intfloat/SimKGC), on which our implements are based.
<!-- Alipay.com Inc.
Copyright (c) 2004-2023 All Rights Reserved. -->
