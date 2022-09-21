#!/bin/zsh
#SBATCH -p alldlc_gpu-rtx2080 #alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -J EVAL_IN_best_config_dino
##SBATCH -t 23:59:59

export PATH="/home/stolld/.conda/bin:$PATH"

pip list

# mkdir -p /work/dlclarge1/stolld-metassl_dino/dino/$EXPERIMENT_NAME/dino_communication
# filename=/work/dlclarge1/stolld-metassl_dino/dino/$EXPERIMENT_NAME/dino_communication/$(openssl rand -hex 12)

python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --data_path /data/datasets/ImageNet/imagenet-pytorch --output_dir /work/dlclarge1/stolld-metassl_dino/dino/ImageNet/$EXPERIMENT_NAME --batch_size_per_gpu 32 --pretrained_weights /work/dlclarge1/stolld-metassl_dino/dino/ImageNet/$EXPERIMENT_NAME/checkpoint.pth --seed $SEED





