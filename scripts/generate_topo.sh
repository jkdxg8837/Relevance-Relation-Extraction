raw_train_txt = "./wn18rr/train.txt"
raw_test_txt = "./wn18rr/test.txt"
# raw_train_txt = "./data/Miniprogram/train.txt"
# raw_test_txt = "./data/Miniprogram/test.txt"
output_dir = "./"

python ./data/Miniprogram/topology/graph_composer_generator.py $raw_train_txt $raw_test_txt $output_dir
python ./data/Miniprogram/topology/node2vec_code.py $output_dir
# Alipay.com Inc.
# Copyright (c) 2004-2023 All Rights Reserved.
