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

We also provide the predictions from our models in [predictions](predictions/) directory.

For WN18RR and FB15k237 datasets, we use files from [KG-BERT](https://github.com/yao8839836/kg-bert).

### WN18RR dataset

Step 1, preprocess the dataset
```
bash scripts/preprocess.sh WN18RR
```

Step 2, training the model and (optionally) specify the output directory (< 3 hours)
```
OUTPUT_DIR=./checkpoint/wn18rr/ bash scripts/train_wn.sh
```

Step 3, evaluate a trained model
```
bash scripts/eval.sh ./checkpoint/wn18rr/model_last.mdl WN18RR
```

Feel free to change the output directory to any path you think appropriate.

### Generating topology info

Unlike the open source dataset, our mini-program & content dataset has clearly defined the level of strength of relevance. So in the topology generation of our dataset, we connect edges except for "no correlation". As for open source datasets, take WN18RR as an example, we measure the semantic meaning of the corresponding 13 relation types, then decides the directions of the edge.
```
bash scripts/generate_topo.sh <dataset name>
```

As this paper is conducted with the help of Ant Group, so we need to follow the rule of the open source code of Ant Group. We are positively tracking the approval process of Ant Group, and our final code will be released soon.


### Acknowledgements
Thanks Wang et.al for their open source code [SimKGC](https://github.com/intfloat/SimKGC), on which our implements are based.
<!-- Alipay.com Inc.
Copyright (c) 2004-2023 All Rights Reserved. -->
