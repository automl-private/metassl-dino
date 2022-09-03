#!/bin/bash
#SBATCH -p mldlc_gpu-rtx2080
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:8
#SBATCH -J IN_PT_DINO_RESNET50
#SBATCH -t 3-23:59:59

pip list

source activate dino

# python -u -m torch.distributed.launch --use_env --nproc_per_node=8 --nnodes 1 main_dino.py --arch resnet50 --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/$EXPERIMENT_NAME --batch_size_per_gpu 32 --saveckp_freq 10 --seed $SEED


python -u -m torch.distributed.launch --use_env --nproc_per_node=8 --nnodes 1 main_dino.py --arch resnet50 --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/$EXPERIMENT_NAME --batch_size_per_gpu 32 --saveckp_freq 10 --seed $SEED
