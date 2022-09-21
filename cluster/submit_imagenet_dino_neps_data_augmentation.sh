#!/bin/bash
#SBATCH -p mldlc_gpu-rtx2080
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:8
#SBATCH -J IN_DA
#SBATCH -t 3-23:59:59
#SBATCH --array 0-24%5

pip list

export PATH="/home/stolld/.conda/bin:$PATH"

mkdir -p /work/dlclarge1/stolld-metassl_dino/dino/ImageNet/$EXPERIMENT_NAME/dino_communication
filename=/work/dlclarge1/stolld-metassl_dino/dino/ImageNet/$EXPERIMENT_NAME/dino_communication/$(openssl rand -hex 12)

python -m torch.distributed.launch --use_env --nproc_per_node=8 --nproc_per_node=1 main_dino.py --config_file_path $filename --arch vit_small --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge1/stolld-metassl_dino/dino/ImageNet/$EXPERIMENT_NAME --batch_size_per_gpu 32 --saveckp_freq 10 --seed $SEED --is_neps_run
