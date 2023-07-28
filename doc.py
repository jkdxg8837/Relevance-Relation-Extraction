'''
Alipay.com Inc.
Copyright (c) 2004-2023 All Rights Reserved.
'''
import os
import json
import torch
import torch.utils.data.dataset
import numpy as np

from typing import Optional, List

from config import args
from triplet import reverse_triplet
from triplet_mask import construct_mask, construct_self_negative_mask
from dict_hub import get_entity_dict, get_link_graph, get_tokenizer
from logger_config import logger

entity_dict = get_entity_dict()
topo_dict = {}
# old is deepwalk2
with open('./g_node2vec.emb', 'r') as f:
    for line in f:
        parts = line.strip().split(' ')
        id = parts[0]
        emb = np.array(parts[-128:]).astype(np.float)
        emb_tensor = torch.tensor(emb)
        topo_dict[id] = emb_tensor

# topo_list = np.load('YOUR TOPO PATH/entity_embedding.npy', allow_pickle=True)
# for _idx in range(len(topo_list)):
#     value = torch.tensor(topo_list[_idx])
#     topo_dict[_idx] = value
    
if args.use_link_graph:
    # make the lazy data loading happen
    get_link_graph()


def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    encoded_inputs = tokenizer(text=text,
                               text_pair=None,
                            #    text_pair=text_pair if text_pair else None,
                               add_special_tokens=True,
                               max_length=args.max_num_tokens,
                               return_token_type_ids=True,
                               truncation=True)
    return encoded_inputs


def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    # a very small fraction of entities in wiki5m do not have name
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity


def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id)
    # avoid label leakage during training
    if not args.is_test:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]
    return ' '.join(entities)


class Example:

    def __init__(self, head_id, relation, tail_id, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation

    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity
    def vectorize(self) -> dict:
        head_desc, tail_desc = self.head_desc, self.tail_desc
        relation_desc = self.relation
        relation_score = 0.0
        relation_type = 0
        neg_relation_list = ['明确强相关','偏强相关','弱相关','不相关']
        if '明确强相关' in relation_desc:
            relation_type = 0
            relation_score = 0.9
        elif '偏强相关' in relation_desc:
            relation_type = 1
            relation_score = 0.7
        elif '弱相关' in relation_desc:
            relation_type = 2
            relation_score = 0.5
        else:
            relation_type = 3
            relation_score = 0.1
            
        neg_relation_list.pop(relation_type)
        

        if args.use_link_graph:
            if len(head_desc.split()) < 20:
                head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
            if len(tail_desc.split()) < 20:
                tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)

        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc)
        hr_encoded_inputs = _custom_tokenize(text=head_text,
                                             text_pair=self.relation)
        hr_neg_inputs_ids = []
        hr_neg_token_type_ids = []
        for _idx in range(len(neg_relation_list)):
            single_hr_encoded_inputs = _custom_tokenize(text=head_text,
                                             text_pair=neg_relation_list[_idx])
            hr_neg_inputs_ids.append(single_hr_encoded_inputs['input_ids'])
            hr_neg_token_type_ids.append(single_hr_encoded_inputs['token_type_ids'])
            
        head_encoded_inputs = _custom_tokenize(text=head_text)

        tail_word = _parse_entity_name(self.tail)
        tail_encoded_inputs = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))
        
        global topo_dict
        if self.head_id in topo_dict.keys():
            head_topo_feature = topo_dict[self.head_id]
        else:
            head_topo_feature = torch.zeros(128)
        if self.tail_id in topo_dict.keys():
            tail_topo_feature = topo_dict[self.tail_id]
        else:
            tail_topo_feature = torch.zeros(128)

        return {'hr_token_ids': hr_encoded_inputs['input_ids'],
                'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
                'hr_neg_token_ids': hr_neg_inputs_ids,
                'hr_neg_token_type_ids': hr_neg_token_type_ids,
                'tail_token_ids': tail_encoded_inputs['input_ids'],
                'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
                'head_token_ids': head_encoded_inputs['input_ids'],
                'head_token_type_ids': head_encoded_inputs['token_type_ids'],
                'relation_type':relation_type,
                'relation_score':relation_score,
                'head_id':self.head_id,
                'tail_id':self.tail_id,
                'head_topo_feature': head_topo_feature,
                'tail_topo_feature': tail_topo_feature,
                'obj': self}


class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, task, examples=None):
        self.path_list = path.split(',')
        self.task = task
        assert all(os.path.exists(path) for path in self.path_list) or examples
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    self.examples = load_data(path)
                else:
                    self.examples.extend(load_data(path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].vectorize()


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))

    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None

    return examples


def collate(batch_data: List[dict]) -> dict:
    hr_token_ids, hr_mask = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    hr_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_type_ids']) for ex in batch_data],
        need_mask=False)

    hr_neg_token_ids_list = []
    hr_neg_token_type_ids_list = []
    for ex in batch_data:
        temp = ex['hr_neg_token_ids']
        for _idx in range(len(temp)):
            hr_neg_token_ids_list.append(torch.LongTensor(temp[_idx]))
    for ex in batch_data:
        temp = ex['hr_neg_token_type_ids']
        for _idx in range(len(temp)):
            hr_neg_token_type_ids_list.append(torch.LongTensor(temp[_idx]))
    hr_neg_token_ids,hr_neg_mask = to_indices_and_mask(hr_neg_token_ids_list, pad_token_id=get_tokenizer().pad_token_id)
    hr_neg_token_type_ids = to_indices_and_mask(hr_neg_token_type_ids_list, need_mask = False)

    head_ids = [ex['head_id'] for ex in batch_data]
    tail_ids = [ex['tail_id'] for ex in batch_data]


    tail_token_ids, tail_mask = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    tail_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data],
        need_mask=False)

    head_token_ids, head_mask = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    head_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_type_ids']) for ex in batch_data],
        need_mask=False)
    
    relation_types = torch.LongTensor([ex['relation_type'] for ex in batch_data])
    relation_scores = torch.FloatTensor([ex['relation_score'] for ex in batch_data])
    head_topo_features_list = [ex['head_topo_feature'] for ex in batch_data]
    head_topo_features = torch.stack(head_topo_features_list, 0)

    tail_topo_features_list = [ex['tail_topo_feature'] for ex in batch_data]
    tail_topo_features = torch.stack(tail_topo_features_list, 0)

    batch_exs = [ex['obj'] for ex in batch_data]
    batch_dict = {
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_mask,
        'hr_token_type_ids': hr_token_type_ids,
        'hr_neg_token_ids':hr_neg_token_ids,
        'hr_neg_mask':hr_neg_mask,
        'hr_neg_token_type_ids':hr_neg_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
        'relation_type': relation_types,
        'relation_score': relation_scores,
        'head_ids': head_ids,
        'tail_ids': tail_ids,
        'head_topo_features': head_topo_features.half(),
        'tail_topo_features': tail_topo_features.half(),
    }

    return batch_dict


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices
