#!/bin/zsh
#SBATCH -p mldlc_gpu-rtx2080 # alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -J dino_neps_imgnet_balanced_data_aug
#SBATCH -t 5-23:59:59  # 23:59:59
#SBATCH --array 0-9999%9

source /home/ferreira/.profile
source activate dino

mkdir -p /work/dlclarge2/ferreira-dino/metassl-dino/experiments/$EXPERIMENT_NAME/dino_communication

filename=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/$EXPERIMENT_NAME/dino_communication/$(openssl rand -hex 12)

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 main_dino.py --config_file_path $filename --arch vit_small --config_space "data_augmentation" --dataset ImageNet --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge2/ferreira-dino/metassl-dino/experiments/$EXPERIMENT_NAME --batch_size_per_gpu 40 --is_neps_run
