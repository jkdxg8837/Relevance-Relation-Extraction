'''
Alipay.com Inc.
Copyright (c) 2004-2023 All Rights Reserved.
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

from time import time
from typing import List, Tuple
from dataclasses import dataclass, asdict
import sys
sys.path.append('./')
from config import args
from doc import load_data, Example
from predict import BertPredictor
from dict_hub import get_entity_dict, get_all_triplet_dict
from triplet import EntityDict
from rerank import rerank_by_graph
from logger_config import logger
from sklearn.metrics import classification_report, coverage_error, ndcg_score, label_ranking_average_precision_score, label_ranking_loss , roc_auc_score

import numpy as np
from sklearn.metrics import classification_report
# EMOS_WN18RR=['_member_of_domain_usage','_has_part','_also_see','_hypernym','_synset_domain_topic_of','_derivationally_related_form','_similar_to','_instance_hypernym','_verb_group','_member_meronym','_member_of_domain_region']
EMOS_WN18RR=['member of domain usage','has part','also see','hypernym','synset domain topic of','derivationally related form','similar to','instance hypernym','verb group','member meronym','member of domain region']
EMOS_Ant=['明确强相关','偏强相关','弱相关','不相关']
classification_report(['truth'], ['generate'], target_names=EMOS_Ant, digits=4, labels=list(range(len(EMOS_Ant))))
def _setup_entity_dict() -> EntityDict:
    if args.task == 'wiki5m_ind':
        return EntityDict(entity_dict_dir=os.path.dirname(args.valid_path),
                          inductive_test_path=args.valid_path)
    return get_entity_dict()


entity_dict = _setup_entity_dict()
all_triplet_dict = get_all_triplet_dict()
visualize_save_dir = None


@dataclass
class PredInfo:
    head: str
    relation: str
    tail: str
    pred_tail: str
    pred_score: float
    topk_score_info: str
    rank: int
    correct: bool


@torch.no_grad()
def compute_metrics(hr_tensor: torch.tensor,
                    entities_tensor: torch.tensor,
                    target: List[int],
                    examples: List[Example],
                    k=3, batch_size=256) -> Tuple:
    assert hr_tensor.size(1) == entities_tensor.size(1)
    total = hr_tensor.size(0)
    entity_cnt = len(entity_dict)
    assert entity_cnt == entities_tensor.size(0)
    target = torch.LongTensor(target).unsqueeze(-1).to(hr_tensor.device)
    topk_scores, topk_indices = [], []
    ranks = []

    mean_rank, mrr, hit1, hit3, hit10 = 0, 0, 0, 0, 0

    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = start + batch_size
        # batch_size * entity_cnt
        batch_score = (torch.mm(hr_tensor[start:end, :], entities_tensor.t())+1)/2.0
        assert entity_cnt == batch_score.size(1)
        batch_target = target[start:end]

        # re-ranking based on topological structure
        rerank_by_graph(batch_score, examples[start:end], entity_dict=entity_dict)

        # filter known triplets
        for idx in range(batch_score.size(0)):
            mask_indices = []
            cur_ex = examples[start + idx]
            gold_neighbor_ids = all_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)
            if len(gold_neighbor_ids) > 10000:
                logger.debug('{} - {} has {} neighbors'.format(cur_ex.head_id, cur_ex.relation, len(gold_neighbor_ids)))
            for e_id in gold_neighbor_ids:
                if e_id == cur_ex.tail_id:
                    continue
                mask_indices.append(entity_dict.entity_to_idx(e_id))
            mask_indices = torch.LongTensor(mask_indices).to(batch_score.device)
            batch_score[idx].index_fill_(0, mask_indices, -1)

        batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)
        target_rank = torch.nonzero(batch_sorted_indices.eq(batch_target).long(), as_tuple=False)
        assert target_rank.size(0) == batch_score.size(0)
        for idx in range(batch_score.size(0)):
            idx_rank = target_rank[idx].tolist()
            assert idx_rank[0] == idx
            cur_rank = idx_rank[1]

            # 0-based -> 1-based
            cur_rank += 1
            mean_rank += cur_rank
            mrr += 1.0 / cur_rank
            hit1 += 1 if cur_rank <= 1 else 0
            hit3 += 1 if cur_rank <= 3 else 0
            hit10 += 1 if cur_rank <= 10 else 0
            ranks.append(cur_rank)

        topk_scores.extend(batch_sorted_score[:, :k].tolist())
        topk_indices.extend(batch_sorted_indices[:, :k].tolist())

    metrics = {'mean_rank': mean_rank, 'mrr': mrr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    metrics = {k: round(v / total, 4) for k, v in metrics.items()}
    assert len(topk_scores) == total
    return topk_scores, topk_indices, metrics, ranks

@torch.no_grad()
def compute_relation_metrics(
    head_tensor: torch.tensor,tail_tensor: torch.tensor, 
                    target_labels: List[int],
                    target_scores: List[float],
                    reg_logits: List[float], 
                    examples: List[Example],
                    k=3, batch_size=256) -> Tuple:
    assert head_tensor.size(1) == tail_tensor.size(1)
    # Visualizing distribution graph during prediction
    total = head_tensor.size(0)
    label0_indices = []
    label1_indices = []
    label2_indices = []
    label3_indices = []
    for _idx in range(len(target_labels)):
        if target_labels[_idx] == 0:
            label0_indices.append(_idx)
        elif target_labels[_idx] == 1:
            label1_indices.append(_idx)
        elif target_labels[_idx] == 2:
            label2_indices.append(_idx)
        else:
            label3_indices.append(_idx)
    label0_indices = torch.LongTensor(label0_indices)
    label1_indices = torch.LongTensor(label1_indices)
    label2_indices = torch.LongTensor(label2_indices)
    label3_indices = torch.LongTensor(label3_indices)
    
    target_labels = torch.LongTensor(target_labels).unsqueeze(-1).to(head_tensor.device)
    target_scores = torch.FloatTensor(target_scores).unsqueeze(-1).to(head_tensor.device)
    reg_scores = torch.cat(reg_logits, dim=0)
    
    batch_target = []
    example_size = len(examples)
    relation_size = 4
    for _idx in range(len(examples)):
        batch_target.append((int)(_idx * relation_size))

    model_prd = np.array([])
    gth = np.array([])
    batch_score = reg_scores.unsqueeze(0).repeat(4,1)
    visualize_score = reg_scores
    label0_score = visualize_score.cpu().gather(dim=0, index=label0_indices).numpy()
    label1_score = visualize_score.cpu().gather(dim=0, index=label1_indices).numpy()
    label2_score = visualize_score.cpu().gather(dim=0, index=label2_indices).numpy()
    label3_score = visualize_score.cpu().gather(dim=0, index=label3_indices).numpy()
    all_score = label0_score
    all_score = np.append(all_score,label1_score,axis=0)
    all_score = np.append(all_score,label2_score,axis=0)
    all_score = np.append(all_score,label3_score,axis=0)
    np.savetxt("temp.txt",all_score)
    if visualize_save_dir is not None:
        plt.hist(label0_score, bins=100, alpha=.7)
        plt.savefig(visualize_save_dir+'_label0.png')
        plt.clf()
        plt.hist(label1_score, bins=100, alpha=.7)
        plt.savefig(visualize_save_dir+'_label1.png')
        plt.clf()
        plt.hist(label2_score, bins=100, alpha=.7)
        plt.savefig(visualize_save_dir+'_label2.png')
        plt.clf()
        plt.hist(label3_score, bins=100, alpha=.7)
        plt.savefig(visualize_save_dir+'_label3.png')
        plt.clf()
    else:
        plt.hist(label0_score, bins=100, alpha=.7)
        plt.savefig('./visualize/label0.png')
        plt.clf()
        plt.hist(label1_score, bins=100, alpha=.7)
        plt.savefig('./visualize/label1.png')
        plt.clf()
        plt.hist(label2_score, bins=100, alpha=.7)
        plt.savefig('./visualize/label2.png')
        plt.clf()
        plt.hist(label3_score, bins=100, alpha=.7)
        plt.savefig('./visualize/label3.png')
        plt.clf()
    # In our experiments, 0.9 / 0.7 / 0.5 / 0.1 is the best threshold towards 4 relations
    batch_score[0] = (batch_score[0] - 0.9).abs()
    batch_score[1] = (batch_score[1] - 0.7).abs()
    batch_score[2] = (batch_score[2] - 0.5).abs()
    batch_score[3] = (batch_score[3] - 0.1).abs()
    
    batch_score_mae = reg_scores
    L1Loss = torch.nn.L1Loss()
    L2Loss = torch.nn.MSELoss()
    target_scores_l1 = L1Loss(batch_score_mae.cpu(), target_scores.squeeze(1).cpu())
    target_scores_l2 = L2Loss(batch_score_mae.cpu(), target_scores.squeeze(1).cpu())
    
    _, prd_labels = batch_score.min(dim = 0)

    label0_indices = torch.from_numpy(np.argwhere(target_labels.squeeze(1).cpu().numpy() < 1))
    label3_indices = torch.from_numpy(np.argwhere(target_labels.squeeze(1).cpu().numpy() > 2))
    report = classification_report(target_labels.squeeze(-1).cpu(), prd_labels.cpu(), target_names=EMOS_Ant, digits=11, labels=list(range(len(EMOS_Ant))))
    metrics = {'l1 distance': target_scores_l1.item(), 'l2 distance': target_scores_l2.item()}
    return report, metrics


def predict_by_split():
    assert os.path.exists(args.valid_path)
    assert os.path.exists(args.train_path)
    args.is_test = True
    predictor = BertPredictor()
    predictor.load(ckt_path=args.eval_model_path)

    relation_tensor ,relation_labels, relation_scores= predictor.predict_by_relations(entity_dict.entity_exs, EMOS_Ant)
    cls_result, forward_metrics = eval_relation(predictor,
                                     relation_tensor=relation_tensor,
                                     target_labels = relation_labels,
                                     target_scores = relation_scores,
                                     eval_forward=True)

    backward_metrics = forward_metrics
    
    metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
    logger.info('Averaged metrics: {}'.format(metrics))

    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    with open('{}/metrics_{}_{}.json'.format(prefix, split, basename), 'w', encoding='utf-8') as writer:
        writer.write('forward metrics: {}\n'.format(json.dumps(forward_metrics)))
        writer.write('backward metrics: {}\n'.format(json.dumps(backward_metrics)))
        writer.write('average metrics: {}\n'.format(json.dumps(metrics)))


def eval_single_direction(predictor: BertPredictor,
                          entity_tensor: torch.tensor,
                          eval_forward=True,
                          batch_size=256) -> dict:
    start_time = time()
    examples = load_data(args.valid_path, add_forward_triplet=eval_forward, add_backward_triplet=not eval_forward)

    hr_tensor, _ = predictor.predict_by_examples(examples)
    hr_tensor = hr_tensor.to(entity_tensor.device)
    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
    logger.info('predict tensor done, compute metrics...')
    # hr_tensor x*768; entity_tensor y*768, x is equal to the size of samples in test split; y indicates the size of all entities.
    topk_scores, topk_indices, metrics, ranks = compute_metrics(hr_tensor=hr_tensor, entities_tensor=entity_tensor,
                                                                target=target, examples=examples,
                                                                batch_size=batch_size)
    eval_dir = 'forward' if eval_forward else 'backward'
    logger.info('{} metrics: {}'.format(eval_dir, json.dumps(metrics)))

    pred_infos = []
    for idx, ex in enumerate(examples):
        cur_topk_scores = topk_scores[idx]
        cur_topk_indices = topk_indices[idx]
        pred_idx = cur_topk_indices[0]
        cur_score_info = {entity_dict.get_entity_by_idx(topk_idx).entity: round(topk_score, 3)
                          for topk_score, topk_idx in zip(cur_topk_scores, cur_topk_indices)}

        pred_info = PredInfo(head=ex.head, relation=ex.relation,
                             tail=ex.tail, pred_tail=entity_dict.get_entity_by_idx(pred_idx).entity,
                             pred_score=round(cur_topk_scores[0], 4),
                             topk_score_info=json.dumps(cur_score_info),
                             rank=ranks[idx],
                             correct=pred_idx == target[idx])
        pred_infos.append(pred_info)

    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    with open('{}/eval_{}_{}_{}.json'.format(prefix, split, eval_dir, basename), 'w', encoding='utf-8') as writer:
        writer.write(json.dumps([asdict(info) for info in pred_infos], ensure_ascii=False, indent=4))

    logger.info('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))
    return metrics

def get_index_to_normalize(sims, videos):
    argm = np.argsort(-sims, axis=1)[:,0]
    result = np.array(list(map(lambda x: x.item() in videos, argm)))
    result = np.nonzero(result)
    return result
def eval_relation(predictor: BertPredictor,
                          relation_tensor: torch.tensor,
                          target_labels: torch.tensor, 
                          target_scores: torch.tensor,
                          eval_forward=True,
                          batch_size=256) -> dict:
    start_time = time()
    examples = load_data(args.valid_path, add_forward_triplet=eval_forward, add_backward_triplet=not eval_forward)
    train_examples_backward = load_data(args.train_path, add_forward_triplet=False, add_backward_triplet=True)
    head_tensor, tail_tensor, reg_logits, head_ids, tail_ids= predictor.predict_by_examples(examples)
    logger.info('predict tensor done, compute metrics...')
    relation_cls, metrics = compute_relation_metrics(head_tensor=head_tensor, tail_tensor=tail_tensor, 
                                                                target_labels=target_labels, target_scores = target_scores, reg_logits = reg_logits, examples=examples,
                                                                batch_size=batch_size)
    print(relation_cls)
    eval_dir = 'forward' if eval_forward else 'backward'
    logger.info('{} metrics: {}'.format(eval_dir, json.dumps(metrics)))

    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)


    logger.info('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))
    return relation_cls, metrics
def predict_by_split_training(eval_path, save_dir):
    args.eval_model_path = eval_path
    global visualize_save_dir
    visualize_save_dir = save_dir
    assert os.path.exists(args.valid_path)
    assert os.path.exists(args.train_path)
    args.is_test = True
    predictor = BertPredictor()
    predictor.load(ckt_path=args.eval_model_path)

    relation_tensor ,relation_labels, relation_scores= predictor.predict_by_relations(entity_dict.entity_exs, EMOS_Ant)
    
    cls_result, forward_metrics = eval_relation(predictor,
                                     relation_tensor=relation_tensor,
                                     target_labels = relation_labels,
                                     target_scores = relation_scores,
                                     eval_forward=True)

    
if __name__ == '__main__':
    predict_by_split()
