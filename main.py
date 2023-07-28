'''
Alipay.com Inc.
Copyright (c) 2004-2023 All Rights Reserved.
'''
import torch
import json
import torch.backends.cudnn as cudnn

from config import args
# For regression relevance score training, use reg_trainer; For ordinary open-source dataset training (e.g. WN18RR), using trainer.py in SimKGC.
from reg_trainer import Trainer
from logger_config import logger


def main():
    ngpus_per_node = torch.cuda.device_count()
    cudnn.benchmark = True

    logger.info("Use {} gpus for training".format(ngpus_per_node))

    trainer = Trainer(args, ngpus_per_node=ngpus_per_node)
    logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    trainer.train_loop()


if __name__ == '__main__':
    main()
