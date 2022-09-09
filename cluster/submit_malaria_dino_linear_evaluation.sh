#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J MAL_EVAL_DINO
#SBATCH -t 00:59:59

pip list

source activate dino


port=`python cluster/find_free_port.py`
echo "found free port $port"

python -m torch.distributed.launch --master_port=$port --nproc_per_node=1 eval_linear.py --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/$EXPERIMENT_NAME --pretrained_weights /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/$EXPERIMENT_NAME/checkpoint.pth --batch_size_per_gpu 256 --gpu 1 --world_size 1 --dataset Malaria --num_labels 2 --epochs 100
