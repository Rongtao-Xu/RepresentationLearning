#!/bin/bash
#SBATCH -A test
#SBATCH -J attn_reg
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH -p short
#SBATCH -t 3-0:00:00
#SBATCH -o wetr_attn_reg.out

source activate py36

port=27978
crop_size=320

file=scripts/dist_train_voc.py
config=configs/voc_attn_reg.yaml



CUDA_VISIBLE_DEVICES=2,3 echo python -m torch.distributed.launch --nproc_per_node=2 --master_port=$port $file --config $config --pooling gmp --crop_size $crop_size --work_dir work_dir_voc
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$port $file --config $config
--pooling gmp --crop_size $crop_size --work_dir work_dir_voc



