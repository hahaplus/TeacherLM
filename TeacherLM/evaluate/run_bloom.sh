#!/bin/sh


python main.py \
    --model bloom \
    --model_args pretrained=${1} \
    --device 0 \
    --batch_size 1 \
    --no_cache \
    --tasks ${2}