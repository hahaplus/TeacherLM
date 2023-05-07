#!/bin/sh
NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export TF_ENABLE_ONEDNN_OPTS=0
# export TORCH_EXTENSIONS_DIR=/tmp/$USER/torch_extensions/
export PYTHONIOENCODING=utf8
#* If your training platform could not access to wandb when training, you can use `offline` mode to log your experiment.
export WANDB_PROJECT=${10}
#export WANDB_MODE="offline"


#! Here is the parameter list and its index:
#! 0 {script} 1 {model_path} 2 {dataset_path / dataset_name} 3 {per_device_batch_size} 4 {accumulation} 5 {learning_rate} 6 {num_train_epochs} 7 {task_type} 8 {store_name / wandb_run_name} 9 {store_dir} 10 {wandb_project}

#* {model_path} : The model's path (with tokenizer in the same position), which you can load through `transformers.AutoModel.from_pretrained`.
#* {dataset_path}: Dataset path, which you can load through `dataset.load_from_disk`.
#* {per_device_batch_size}: Batch size on each node.
#* {accumulation}: When GPU memory is insufficient, you can train by adjusting the value of `--gradient_accumulation_steps`. Then, your total batch_size is per_device_batch_size * accumulation * GPU_nums.
#* {learning_rate}: Set learning rate, with `--lr constant` can set constant learning rate.
#* {num_train_epochs}: Set train epochs.
#* {task_type}: The field in your provided dataset you want to train on.
#* {store_name / wandb_run_name}: Your checkpoints will be saved at {store_dir}/{store_name}, whereas your wandb_run_name will also be set to {store_name}.
#* {store_dir}: The root directory where you want to save your checkpoints.
#* {wandb_project}: The wandb project name you want to log your experiment.

CMD="torchrun --nnodes ${NUM_WORKERS} --nproc_per_node ${NUM_GPUS_PER_WORKER} finetune.py"
CMD+="  --model_name_or_path ${1}"
CMD+="  --dataset_name ${2}"
CMD+="  --per_device_train_batch_size ${3}"
CMD+="  --gradient_checkpointing true --gradient_accumulation_steps ${4}"
CMD+="  --learning_rate ${5} --lr constant"
CMD+="  --do_train --num_train_epochs ${6}"
CMD+="  --task_type ${7}"
CMD+="  --max_source_length 2048"
CMD+="  --max_target_length 2048"
CMD+="  --run_name ${8}"
CMD+="  --output_dir ${9}/${8}"
CMD+="  --save_strategy epoch"
CMD+="  --logging_steps 8"
CMD+="  --deepspeed ./deepspeed_config.json"
CMD+="  --fp16"
CMD+="  --report_to wandb"

echo $CMD
eval $CMD