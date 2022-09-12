#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J C100_PT_DINO
##SBATCH -t 23:59:59

pip list

source activate dino

port=`python cluster/find_free_port.py`
echo "found free port $port"

# Baseline
# python -m torch.distributed.launch --master_port=$port --nproc_per_node=1 main_dino.py --arch vit_small --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/$EXPERIMENT_NAME --batch_size_per_gpu 256 --saveckp_freq 10 --gpu 1 --world_size 1 --dataset CIFAR-100 --epochs 800 --seed $SEED

# Best data augmentation conf
# python -m torch.distributed.launch --master_port=$port --nproc_per_node=1 main_dino.py --arch vit_small --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/$EXPERIMENT_NAME --batch_size_per_gpu 256 --saveckp_freq 10 --gpu 1 --world_size 1 --dataset CIFAR-100 --epochs 800 --seed $SEED --use_fixed_DA_hypers --local_crops_number 8

# Best training conf
python -m torch.distributed.launch --master_port=$port --nproc_per_node=1 main_dino.py --arch vit_small --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/$EXPERIMENT_NAME --batch_size_per_gpu 256 --saveckp_freq 10 --gpu 1 --world_size 1 --dataset CIFAR-100 --epochs 800 --seed $SEED \
	--lr 0.0008319 \
	--out_dim 51866 \
	--momentum_teacher 0.9697879 \
	--warmup_teacher_temp 0.0568459 \
	--warmup_teacher_temp_epochs 6 \
	--weight_decay 0.2319290 \
	--weight_decay_end 0.4093431 \
	--freeze_last_layer 1 \
	--warmup_epochs 31 \
	--min_lr 0.0000050 \
	--drop_path_rate 0.0850883 \
	--optimizer adamw
