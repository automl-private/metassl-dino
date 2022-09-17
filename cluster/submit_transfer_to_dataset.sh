#!/bin/bash
#SBATCH -p testdlc_gpu-rtx2080
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:8  # Take only 1 GPU if not running for iNaturalists
#SBATCH -J Transfer_EVAL_DINO
##SBATCH -t 23:59:59

source activate dino

pip list

port=`python cluster/find_free_port.py`
echo "found free port $port"

# iNaturalists (1 node)
python -m torch.distributed.launch --master_port=$port --nproc_per_node=8 --nnodes=1 eval_linear.py --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/$DATASET/$EXPERIMENT_NAME --pretrained_weights /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/22-08-05_baseline_vit_seed$SEED/checkpoint.pth --seed $SEED --dataset $DATASET --batch_size_per_gpu 32

# Other datasets (1 GPU)
# python -m torch.distributed.launch --master_port=$port --nproc_per_node=1 --nnodes=1 eval_linear.py --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/$DATASET/$EXPERIMENT_NAME --pretrained_weights /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/22-08-05_baseline_vit_seed$SEED/checkpoint.pth --seed $SEED --dataset $DATASET --batch_size_per_gpu 32
