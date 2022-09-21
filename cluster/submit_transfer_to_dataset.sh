#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --gres=gpu:1  # if iNaturalists: 8, else: 1
#SBATCH -J Transfer_EVAL_DINO
#SBATCH -t 00:59:59

export PATH="/home/stolld/.conda/bin:$PATH"

pip list

port=`python cluster/find_free_port.py`
echo "found free port $port"

# iNaturalists (1 node)
# python -m torch.distributed.launch --master_port=$port --nproc_per_node=8 --nnodes=1 eval_linear.py --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/$DATASET/$EXPERIMENT_NAME --pretrained_weights /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/22-08-05_baseline_vit_seed$SEED/checkpoint.pth --seed $SEED --dataset $DATASET --batch_size_per_gpu 32


# Other datasets (1 GPU) - Baseline
# python -m torch.distributed.launch --master_port=$port --nproc_per_node=1 --nnodes=1 eval_linear.py --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/$DATASET/$EXPERIMENT_NAME --pretrained_weights /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/22-08-05_baseline_vit_seed$SEED/checkpoint.pth --seed $SEED --dataset $DATASET --batch_size_per_gpu 32


# Other datasets (1 GPU) - config 29
python -m torch.distributed.launch --master_port=$port --nproc_per_node=1 --nnodes=1 eval_linear.py --output_dir /work/dlclarge1/stolld-metassl_dino/dino/$DATASET/$EXPERIMENT_NAME --pretrained_weights /work/dlclarge1/stolld-metassl_dino/dino/ImageNet/22-09-15_config_29_pt+eval_seed$SEED/checkpoint.pth --seed $SEED --dataset $DATASET --batch_size_per_gpu 32
