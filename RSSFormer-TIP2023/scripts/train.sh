#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=2,3
NUM_GPUS=2
export PYTHONPATH=$PYTHONPATH:`pwd`

config_path='baseline.hrnetw32'
model_dir='./log/normal_baseline/hrfxt_finnal_b'

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9696 train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    train.eval_interval_epoch 20
