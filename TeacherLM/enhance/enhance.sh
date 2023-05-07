#!/bin/bash

export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=16
GPUS_PER_NODE=1
NNODES=1

export LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    "

#! enhance.sh {TLM path} {input file path} {output file path} {task_type}

export CMD="enhance.py \
    --name ${1}\
    --batch_size 1\
    --input ${2} \
    --output ${3} \
    --task_type ${4}\
"

bash -c '$LAUNCHER $CMD'
