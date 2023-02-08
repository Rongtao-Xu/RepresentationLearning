#!/bin/bash
#SBATCH -A test
#SBATCH -J attn_coco
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH -p short
#SBATCH -t 3-0:00:00
#SBATCH -o wetr_attn_coco.out

source activate py36

port=29171
crop_size=512

file=scripts/dist_train_coco.py
config=configs/coco_attn_reg.yaml




CUDA_VISIBLE_DEVICES=2 echo python -m torch.distributed.launch --nproc_per_node=1 --master_port=$port $file --config $config --pooling gmp --crop_size $crop_size --work_dir work_dir_attn_coco
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=$port $file --config $config --pooling gmp --crop_size $crop_size --work_dir work_dir_attn_coco
