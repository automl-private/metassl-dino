#!/bin/bash
#SBATCH -p mldlc_gpu-rtx2080 #alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -J IN_dino_full_train_fixed_da_hypers_c29_s125
#SBATCH -t 3-23:59:59
##SBATCH -t 23:59:59

export PATH="/home/stolld/.conda/bin:$PATH"

pip list

mkdir -p /work/dlclarge1/stolld-metassl_dino/dino/ImageNet/$EXPERIMENT_NAME/dino_communication
filename=/work/dlclarge1/stolld-metassl_dino/dino/ImageNet/$EXPERIMENT_NAME/dino_communication/$(openssl rand -hex 12)

# Best data augmentation conf
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 main_dino.py --config_file_path $filename --arch vit_small --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge1/stolld-metassl_dino/dino/ImageNet/$EXPERIMENT_NAME --batch_size_per_gpu 32 --dataset ImageNet --seed $SEED --use_fixed_DA_hypers --local_crops_number 8
