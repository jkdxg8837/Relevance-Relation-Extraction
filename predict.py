'''
Alipay.com Inc.
Copyright (c) 2004-2023 All Rights Reserved.
'''
import os
import json
import tqdm
import torch
import torch.utils.data
import numpy as np

import sys
sys.path.append('./')

from typing import List
from collections import OrderedDict
from doc import collate, Example, Dataset, load_data
from config import args
from models_no_hr_vector import build_reg_model
from utils import AttrDict, move_to_cuda, get_model_obj
from dict_hub import build_tokenizer
from logger_config import logger

class BertPredictor:

    def __init__(self):
        self.model = None
        self.train_args = AttrDict()
        self.use_cuda = False

    def load(self, ckt_path, use_data_parallel=False):
        assert os.path.exists(ckt_path)
        ckt_dict = torch.load(ckt_path, map_location=lambda storage, loc: storage)
        self.train_args.__dict__ = ckt_dict['args']
        self._setup_args()
        build_tokenizer(self.train_args)
        self.model = build_reg_model(self.train_args)

        # DataParallel will introduce 'module.' prefix
        state_dict = ckt_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.eval()

        if use_data_parallel and torch.cuda.device_count() > 1:
            logger.info('Use data parallel predictor')
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.use_cuda = True
        elif torch.cuda.is_available():
            self.model.cuda()
            self.use_cuda = True
        logger.info('Load model from {} successfully'.format(ckt_path))

    def _setup_args(self):
        for k, v in args.__dict__.items():
            if k not in self.train_args.__dict__:
                logger.info('Set default attribute: {}={}'.format(k, v))
                self.train_args.__dict__[k] = v
        logger.info('Args used in training: {}'.format(json.dumps(self.train_args.__dict__, ensure_ascii=False, indent=4)))
        args.use_link_graph = self.train_args.use_link_graph
        args.is_test = True

    @torch.no_grad()
    def predict_by_examples(self, examples: List[Example]):
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task),
            num_workers=1,
            batch_size=max(args.batch_size, 1024),
            collate_fn=collate,
            shuffle=False)

        hr_tensor_list, tail_tensor_list, reg_logits_list= [], [], []
        head_id_list, tail_id_list = [], []
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader)):
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
            hr_tensor_list.append(outputs['head_vector'])
            tail_tensor_list.append(outputs['tail_vector'])
            reg_logits_list.append(outputs['reg_logits'])
            
            head_id_list = head_id_list + batch_dict['head_ids']
            tail_id_list = tail_id_list + batch_dict['tail_ids']
            

        return torch.cat(hr_tensor_list, dim=0), torch.cat(tail_tensor_list, dim=0), reg_logits_list, head_id_list, tail_id_list
    @torch.no_grad()
    def predict_by_entities(self, entity_exs) -> torch.tensor:
        examples = []
        for entity_ex in entity_exs:
            examples.append(Example(head_id='', relation='',
                                    tail_id=entity_ex.entity_id))
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task),
            num_workers=2,
            batch_size=max(args.batch_size, 1024),
            collate_fn=collate,
            shuffle=False)

        ent_tensor_list = []
        entity_id_list = []
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader)):
            batch_dict['only_ent_embedding'] = True
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            entity_id_list = entity_id_list + batch_dict['tail_ids']
            ent_tensor_list.append(outputs['ent_vectors'])

        return torch.cat(ent_tensor_list, dim=0), entity_id_list
    @torch.no_grad()
    def predict_by_relations(self, entity_exs, relation_list) -> torch.tensor:
        examples = load_data(args.valid_path, add_forward_triplet=True, add_backward_triplet=not True)
        examples_corrupte_relation = []
        relation_gth_list = []
        relation_score_list = []
        for _idx in range(len(examples)):
            relation = examples[_idx].relation
            # examples_corrupte_relation.append(examples[_idx])
            if '明确强相关' in relation:
                relation_gth = 0
                relation_score_gth = 0.9
            elif '偏强相关' in relation:
                relation_gth = 1
                relation_score_gth = 0.7
            elif '弱相关' in relation:
                relation_gth = 2
                relation_score_gth = 0.5
            else:
                relation_gth = 3
                relation_score_gth = 0.1
            relation_gth_list.append(relation_gth)
            relation_score_list.append(relation_score_gth)
            for _r_idx in range(len(relation_list)):
                # if relation_list[_r_idx] != relation:
                import copy
                temp = copy.deepcopy(examples[_idx])
                temp.relation = relation_list[_r_idx]
                examples_corrupte_relation.append(temp)
                del temp
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples_corrupte_relation, task=args.task),
            num_workers=2,
            batch_size=max(args.batch_size, 1024),
            collate_fn=collate,
            shuffle=False)
        return None, relation_gth_list, relation_score_list
