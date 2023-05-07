#!/bin/sh

python main.py \
    --model opt \
    --model_args pretrained=${1} \
    --device 0 \
    --batch_size 5 \
    --no_cache \
    --tasks ${2}