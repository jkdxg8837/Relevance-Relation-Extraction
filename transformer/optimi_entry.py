'''
Alipay.com Inc.
Copyright (c) 2004-2023 All Rights Reserved.
'''
from torch.optim import Adam



def build_optimizer(model):
    params_enc = []
    params_dec = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            params_enc.append(param)
        else:
            params_dec.append(param)
    all_params = [{'params':params_enc, 'lr': 0.1},
                  {'params':params_dec, 'lr': 1}]
    optimizer = Adam(all_params, lr=1, betas=(0.9, 0.98))
    return optimizer


