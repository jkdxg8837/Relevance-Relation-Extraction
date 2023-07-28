#!/usr/bin/env bash

set -x
set -e

TASK="MINIPROGRAM"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi
echo $DATA_DIR
python -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model "YOUR LANGUAGE MODEL CHECKPOINT PATH" \
--pooling mean \
--lr 8e-6 \
--use-link-graph \
--train-path "YOUR TRAIN PATH" \
--valid-path "YOUR VALID PATH" \
--task ${TASK} \
--batch-size 128 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--pre-batch 0 \
--finetune-t \
--epochs 50 \
--workers 4 \
--max-to-keep 3 "$@"
# Alipay.com Inc.
# Copyright (c) 2004-2023 All Rights Reserved.