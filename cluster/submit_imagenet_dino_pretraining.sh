#!/bin/bash
#SBATCH -p mldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -J IN_PT_DINO_ViT
#SBATCH -t 2-00:00:00

pip list

source activate dino

echo $(date)

# Baseline
# python -u -m torch.distributed.launch --use_env --nproc_per_node=8 --nnodes 1 main_dino.py --arch vit_small --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/$EXPERIMENT_NAME --batch_size_per_gpu 32 --saveckp_freq 10 --seed $SEED \
# 	--train_dataset_percentage_usage 1.0 --valid_size 0.0 --local_crops_number 2 \
	# --batches_per_optimization_step 4

# Baseline with 10% of classes
python -u -m torch.distributed.launch --use_env --nproc_per_node=8 --nnodes 1 main_dino.py --arch vit_small --data_path /work/dlclarge2/wagnerd-metassl-experiments/datasets/ImageNetSubset/10percent/train --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/$EXPERIMENT_NAME --batch_size_per_gpu 32 --saveckp_freq 10 --seed $SEED \
        --train_dataset_percentage_usage 1.0 --valid_size 0.0 --local_crops_number 8 \
        --batches_per_optimization_step 4

# Best DA config 23 or Best DA config 17 (backup)
# python -u -m torch.distributed.launch --use_env --nproc_per_node=8 --nnodes 1 main_dino.py --arch vit_small --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/$EXPERIMENT_NAME --batch_size_per_gpu 32 --saveckp_freq 10 --seed $SEED --use_fixed_DA_hypers --local_crops_number 8

# Best TR config out of 5
# python -u -m torch.distributed.launch --use_env --nproc_per_node=8 --nnodes 1 main_dino.py --arch vit_small --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/$EXPERIMENT_NAME --batch_size_per_gpu 32 --saveckp_freq 10 --seed $SEED \
#        --lr 0.0012897 \
#        --out_dim 46368 \
#        --momentum_teacher 0.9699331 \
#        --warmup_teacher_temp 0.0372770 \
#        --warmup_teacher_temp_epochs 9 \
#        --weight_decay 0.0450683 \
#        --weight_decay_end 0.1693091 \
#        --freeze_last_layer 0 \
#        --warmup_epochs 8 \
#        --min_lr 0.0000033 \
#        --drop_path_rate 0.3841327 \
#        --optimizer adamw

echo $(date)
