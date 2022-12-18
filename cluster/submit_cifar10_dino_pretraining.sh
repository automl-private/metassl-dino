#!/bin/bash
#SBATCH -p testdlc_gpu-rtx2080
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J C10_PT_DINO
##SBATCH -t 23:59:59

pip list

source activate dino

port=`python cluster/find_free_port.py`
echo "found free port $port"

# Baseline
# python -m torch.distributed.launch --master_port=$port --nproc_per_node=1 main_dino.py --arch vit_small --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/$EXPERIMENT_NAME --batch_size_per_gpu 256 --saveckp_freq 10 --gpu 1 --world_size 1 --dataset CIFAR-10 --epochs 800 --seed $SEED

# GroupAugment
python -m torch.distributed.launch --master_port=$port --nproc_per_node=1 main_dino.py --arch vit_small --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/$EXPERIMENT_NAME --batch_size_per_gpu 256 --saveckp_freq 10 --gpu 1 --world_size 1 --dataset CIFAR-10 --epochs 800 --seed $SEED \
	--config_space groupaugment

# Best data augmentation conf
# python -m torch.distributed.launch --master_port=$port --nproc_per_node=1 main_dino.py --arch vit_small --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/$EXPERIMENT_NAME --batch_size_per_gpu 256 --saveckp_freq 10 --gpu 1 --world_size 1 --dataset CIFAR-10 --epochs 800 --seed $SEED --use_fixed_DA_hypers --local_crops_number 5

# Best training conf
# python -m torch.distributed.launch --master_port=$port --nproc_per_node=1 main_dino.py --arch vit_small --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/$EXPERIMENT_NAME --batch_size_per_gpu 256 --saveckp_freq 10 --gpu 1 --world_size 1 --dataset CIFAR-10 --epochs 800 --seed $SEED \
#	--lr 0.0005256 \
#        --out_dim 66796 \
#        --momentum_teacher 0.9952331 \
#        --warmup_teacher_temp 0.0406307 \
#        --warmup_teacher_temp_epochs 16 \
#        --weight_decay 0.0391347 \
#        --weight_decay_end 0.4813455 \
#        --freeze_last_layer 0 \
#        --warmup_epochs 24 \
#        --min_lr 0.0000008 \
#        --drop_path_rate 0.1242921 \
#        --optimizer adamw
