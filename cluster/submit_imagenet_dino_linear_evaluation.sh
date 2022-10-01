#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080 #mldlc_gpu-rtx2080 #bosch_gpu-rtx2080 #alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -J IN_EVAL_DINO_C29_BS40
#SBATCH -t 23:59:59

pip list

source /home/ferreira/.profile
source activate dino

# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 eval_linear.py --data_path /data/datasets/ImageNet/imagenet-pytorch --output_dir /work/dlclarge2/ferreira-dino/metassl-dino/experiments/$EXPERIMENT_NAME --pretrained_weights /work/dlclarge2/ferreira-dino/metassl-dino/experiments/neps_imagenet_balanced_val_data_augmentation/results/config_29/checkpoint.pth --seed $SEED --dataset ImageNet --batch_size_per_gpu 40

# From PT with same experiment name
python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --data_path /data/datasets/ImageNet/imagenet-pytorch --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/$EXPERIMENT_NAME --batch_size_per_gpu 32 --pretrained_weights /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/$EXPERIMENT_NAME/checkpoint.pth --seed $SEED

# From NePS run
# python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --data_path /data/datasets/ImageNet/imagenet-pytorch --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/$EXPERIMENT_NAME --batch_size_per_gpu 32 --pretrained_weights /work/dlclarge2/ferreira-dino/metassl-dino/experiments/neps_imagenet_balanced_val_data_augmentation_new_fixed_balanced_val_set/results/config_4/checkpoint.pth --seed $SEED
