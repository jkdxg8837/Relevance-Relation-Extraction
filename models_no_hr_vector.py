'''
Alipay.com Inc.
Copyright (c) 2004-2023 All Rights Reserved.
'''
from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn
from scipy.stats import spearmanr
from dataclasses import dataclass
from transformers import AutoModel, AutoConfig

from triplet_mask import construct_mask
from transformer.encoders import CrossAtt, SelfAtt

def build_reg_model(args) -> nn.Module:
    return CustomBertModel_Reg(args)

@dataclass
class ModelOutput:
    logits: torch.tensor
    logits_t: torch.tensor
    reg_logits: torch.tensor
    labels: torch.tensor
    reg_labels: torch.tensor
    inv_t: torch.tensor
    head_vector: torch.tensor
    tail_vector: torch.tensor




def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector


class CustomBertModel_Reg(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        self.topo_emb_dim = 128
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]
        # Momentum update factor
        self.m = 0.999
        self.momentum_cl_learning = True
        self.topo_feature_trigger = True
        
    
        self.head_bert = AutoModel.from_pretrained(args.pretrained_model)
        args.pretrained_head_bert = None
        if not args.is_test:
            if args.pretrained_head_bert != None:
                ckt_dict = torch.load(args.pretrained_head_bert, map_location=lambda storage, loc: storage)
                state_dict = ckt_dict['state_dict']
                self.head_bert.load_state_dict(state_dict, strict=True)
        self.momentum_head_bert = deepcopy(self.head_bert)
        
        if self.topo_feature_trigger:
            self.topo_feature_proj = nn.Linear(self.topo_emb_dim,self.topo_emb_dim)
            self.topo_self_att = SelfAtt(d_model = 768+self.topo_emb_dim, N = 3,padding_idx = 0)
            random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
            self.register_buffer("pre_batch_vectors",
                                nn.functional.normalize(random_vector, dim=1),
                                persistent=False)
    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output
    
    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                relation_type,hr_neg_token_ids, hr_neg_token_type_ids, hr_neg_mask,head_topo_features, tail_topo_features,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids,
                                              tail_topo_features = tail_topo_features)
        if self.topo_feature_trigger:
            if self.topo_feature_proj.weight.dtype == torch.float32:
                head_topo_features = head_topo_features.float()
                tail_topo_features = tail_topo_features.float()
            head_topo_features = self.topo_feature_proj(head_topo_features)
            tail_topo_features = self.topo_feature_proj(tail_topo_features)
            
            
            tail_vector = self._encode(self.head_bert,
                                    token_ids=tail_token_ids,
                                    mask=tail_mask,
                                    token_type_ids=tail_token_type_ids)

            head_vector = self._encode(self.head_bert,
                                    token_ids=head_token_ids,
                                    mask=head_mask,
                                    token_type_ids=head_token_type_ids)
            head_topo_features = head_topo_features.to(head_vector.device)
            tail_topo_features = tail_topo_features.to(tail_vector.device)
            
            head_vector ,_= self.topo_self_att(torch.cat((head_vector, head_topo_features), dim=1).to(head_vector.device).unsqueeze(1))
            tail_vector ,_= self.topo_self_att(torch.cat((tail_vector, tail_topo_features), dim=1).to(head_vector.device).unsqueeze(1))
            head_vector = head_vector.squeeze(1)
            tail_vector = tail_vector.squeeze(1)
            
            head_vector = nn.functional.normalize(head_vector, dim=1)
            tail_vector = nn.functional.normalize(tail_vector, dim=1)
            
            if self.momentum_cl_learning:
                with torch.no_grad():
                self._momentum_update_encoder1(self.head_bert, self.momentum_head_bert)
                neg_tail_vector = self._encode(self.momentum_head_bert,
                                        token_ids=tail_token_ids,
                                        mask=tail_mask,
                                        token_type_ids=tail_token_type_ids)
                neg_tail_vector , _= self.topo_self_att(torch.cat((neg_tail_vector, tail_topo_features), dim=1).to(head_vector.device).unsqueeze(1))
                neg_tail_vector = neg_tail_vector.squeeze(1)
                neg_tail_vector = nn.functional.normalize(neg_tail_vector, dim=1)
                    
            else:
                neg_tail_vector = None
            return {'head_vector': head_vector, 'tail_vector': tail_vector, 'neg_tail_vector':neg_tail_vector}
        
        else:
            tail_vector = self._encode(self.head_bert,
                                    token_ids=tail_token_ids,
                                    mask=tail_mask,
                                    token_type_ids=tail_token_type_ids)

            head_vector = self._encode(self.head_bert,
                                    token_ids=head_token_ids,
                                    mask=head_mask,
                                    token_type_ids=head_token_type_ids)
            if self.momentum_cl_learning:
                with torch.no_grad():
                    self._momentum_update_encoder1(self.head_bert, self.momentum_head_bert)
                    neg_tail_vector = self._encode(self.momentum_head_bert,
                                            token_ids=tail_token_ids,
                                            mask=tail_mask,
                                            token_type_ids=tail_token_type_ids)
            else:
                neg_tail_vector = None
            return {'head_vector': head_vector, 'tail_vector': tail_vector, 'neg_tail_vector':neg_tail_vector}
        # DataParallel only support tensor/dict
        # return {'head_vector': head_vector, 'hr_vector': hr_vector,'tail_vector': tail_vector, 'hr_self_neg_vector':hr_self_neg_vector}
    @torch.no_grad()
    def _momentum_update_encoder1(self, encoder1, encoder2):
        """
        Momentum update of the key encoder
        """
        for param_1, param_2 in zip(encoder1.parameters(), encoder2.parameters()):
            param_2.data = param_2.data * self.m + param_1.data * (1. - self.m)
        
    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        head_vector, tail_vector = output_dict['head_vector'], output_dict['tail_vector']
        neg_tail_vector = output_dict['neg_tail_vector']
        
        
        batch_size = head_vector.size(0)
        labels = torch.arange(batch_size).to(head_vector.device)

        logits = (head_vector.mm(tail_vector.t())+1) / 2.0
        logits_t = ((head_vector.mm(tail_vector.t())+1) / 2.0).t()
        relation_score_label = batch_dict['relation_score'].unsqueeze(-1).repeat(1, logits.size(1))
        logits = -torch.abs(logits-relation_score_label) + 1
        logits_t = -torch.abs(logits_t-relation_score_label) + 1
        logits_t = logits_t.t()
        positive_logits = torch.diag_embed(logits.diag())
        
        if self.momentum_cl_learning:
            neg_logits = (head_vector.mm(neg_tail_vector.t())+1) / 2.0
            neg_logits = -torch.abs(neg_logits-relation_score_label) + 1
            neg_logits_diag = neg_logits.diag()
            neg_logits_diag = torch.diag_embed(neg_logits_diag)
            logits = neg_logits - neg_logits_diag + positive_logits
        reg_logits_prd = ((head_vector.mm(tail_vector.t())+1)/2.0).diag()
        
        
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()
        # 防止正确的triplets被选为负样本
        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            if not self.momentum_cl_learning:
                pre_batch_logits = self._compute_pre_batch_logits(head_vector, tail_vector, batch_dict)
                logits = torch.cat([logits, pre_batch_logits], dim=-1)
            else:
                pre_batch_logits = self._compute_pre_batch_logits(head_vector, neg_tail_vector, batch_dict)
                logits = torch.cat([logits, pre_batch_logits], dim=-1)
        return {'logits': logits,
                'logits_t': logits_t,
                'reg_logits': reg_logits_prd,
                'labels': labels,
                'reg_labels': batch_dict['relation_score'],
                'inv_t': self.log_inv_t.detach().exp(),
                'head_vector': head_vector.detach(),
                'tail_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, tail_topo_features,**kwargs) -> dict:
        if self.topo_feature_trigger:
            ent_vectors = self._encode(self.head_bert,
                                    token_ids=tail_token_ids,
                                    mask=tail_mask,
                                    token_type_ids=tail_token_type_ids)
        
            
            ent_vectors ,_= self.topo_self_att(torch.cat((ent_vectors, tail_topo_features), dim=1).to(ent_vectors.device).unsqueeze(1))

            ent_vectors = ent_vectors.squeeze(1)

            
            ent_vectors = nn.functional.normalize(ent_vectors, dim=1)
        else:
            ent_vectors = self._encode(self.head_bert,
                                    token_ids=tail_token_ids,
                                    mask=tail_mask,
                                    token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}
