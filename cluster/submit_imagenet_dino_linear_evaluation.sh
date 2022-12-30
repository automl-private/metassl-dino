#!/bin/bash
#SBATCH -p mldlc_gpu-rtx2080 #mldlc_gpu-rtx2080 #bosch_gpu-rtx2080 #alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -J IN_EVAL_TV2_DINO_Baseline
#SBATCH -t 23:59:59

pip list

source activate dino

# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 eval_linear.py --data_path /data/datasets/ImageNet/imagenet-pytorch --output_dir /work/dlclarge2/ferreira-dino/metassl-dino/experiments/$EXPERIMENT_NAME --pretrained_weights /work/dlclarge2/ferreira-dino/metassl-dino/experiments/neps_imagenet_balanced_val_data_augmentation/results/config_29/checkpoint.pth --seed $SEED --dataset ImageNet --batch_size_per_gpu 40

# From PT with same experiment name
python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --data_path /data/datasets/ImageNet/imagenet-pytorch --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/$EXPERIMENT_NAME --batch_size_per_gpu 32 --pretrained_weights /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/$EXPERIMENT_NAME/checkpoint.pth --seed $SEED \
# 	--train_dataset_percentage_usage 0.1 --valid_size 0.1

# From PT with same experiment name (10% of classes)
# python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --data_path /work/dlclarge2/wagnerd-metassl-experiments/datasets/ImageNetSubset/10percent/ --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/$EXPERIMENT_NAME --batch_size_per_gpu 32 --pretrained_weights /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/$EXPERIMENT_NAME/checkpoint.pth --seed $SEED \
# 	--train_dataset_percentage_usage 0.1 --valid_size 0.1

# From NePS run
# python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --data_path /data/datasets/ImageNet/imagenet-pytorch --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/$EXPERIMENT_NAME --batch_size_per_gpu 32 --pretrained_weights /work/dlclarge2/ferreira-dino/metassl-dino/experiments/neps_imagenet_balanced_val_data_augmentation_new_fixed_balanced_val_set_backup_31_Oct_2022/results/config_23/checkpoint.pth --seed $SEED

# From Baseline
# python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --data_path /data/datasets/ImageNet/imagenet-pytorch --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/$EXPERIMENT_NAME --batch_size_per_gpu 32 --pretrained_weights /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/22-08-05_baseline_vit_seed0/checkpoint.pth --seed $SEED

# From DA config
# python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --data_path /data/datasets/ImageNet/imagenet-pytorch --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/$EXPERIMENT_NAME --batch_size_per_gpu 32 --pretrained_weights /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/22-11-28_DA_config23_pre-training_seed0/checkpoint.pth --seed $SEED

# From Baseline - val
# python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --data_path /data/datasets/ImageNet/imagenet-pytorch --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/$EXPERIMENT_NAME --batch_size_per_gpu 32 --pretrained_weights /work/dlclarge1/stolld-metassl_dino/dino/ImageNet/22-09-22_baseline_balanced_val_pt+eval_seed0/checkpoint.pth --seed $SEED
