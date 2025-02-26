# RRE: A Relevance Relation Extraction Framework for Cross-Domain Recommender System at Alipay

This is a reference code for the following paper presented at IEEE ICME 2024. 

> Jiayang Gu, Xovee Xu, Yulu Tian, Yurun Hu, Jiadong Huang, Wenliang Zhong, Fan Zhou, Lianli Gao  
> RRE: A Relevance Relation Extraction Framework for Cross-Domain Recommender System at Alipay  
> IEEE International Conference on Multimedia and Expo (ICME), Niagara Falls, Canada, July 15-19, 2024, 1-6.


## How to Run

### Requirement

- `python>=3.7`
- `torch>=1.6`
- `transformers>=4.15`
- `node2vec`


### Running

The running or our code involves 3 steps: (1) data processing; (2) model training; and (3) evaluation. 

The predictions of our models are in [predictions](predictions/) directory. The WN18RR and FB15k-237 datasets are from the [KG-BERT](https://github.com/yao8839836/kg-bert) repo. 

Take WN18RR as an example:

- **Step 1**: Data preprocessing  
`bash scripts/preprocess.sh WN18RR`
- **Step 2**: Generate topological information  
Unlike the open source dataset, our mini-program & content dataset has clearly defined the level of strength of relevance. So in the topology generation of our dataset, we connect edges except for "no correlation". As for open source datasets, take WN18RR as an example, we measure the semantic meaning of the corresponding 13 relation types, then decides the directions of the edge.  
`bash scripts/generate_topo.sh`  
By replacing the right path of your train.txt and test.txt of WN18RR, you can get an embedding file under output path.
- **Step 3**: Model training (<3 hours)  
`OUTPUT_DIR=./checkpoint/wn18rr/ bash scripts/train_wn.sh`
- **Step 4**: Evaluation (on a trained model)  
`bash scripts/eval.sh ./checkpoint/wn18rr/model_last.mdl WN18RR`

Also, there exist some triggers to control modules used in our model, located in `models_no_hr_vector.py`. If you don't want to use momentum learning or topology info, feel free to set `FALSE` to them.

## Citation

```bibtex
@inproceedings{gu2024rre,
  title = {RRE: A Relevance Relation Extraction Framework for Cross-Domain Recommender System at Alipay}
  author = {Jiayang Gu and Xovee Xu and Yulu Tian and Yurun Hu and Jiadong Huang and Wenliang Zhong and Fan Zhou and Lianli Gao}, 
  booktitle = {IEEE International Conference on Multimedia and Expo}, 
  year = {2024},
  pages = {1--6},
  address = {Niagara Falls, Canada}, 
  publisher = {IEEE}
}
```

## Acknowledgements

We would like to thank Wang et al. for their open source code [SimKGC](https://github.com/intfloat/SimKGC), on which our implements are based. We would also like to thank ICME '24 Program Committee members for selecting our paper as one of the fifteen Best Paper Candidates.
