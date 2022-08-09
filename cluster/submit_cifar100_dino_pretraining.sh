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

python -m torch.distributed.launch --master_port=$port --nproc_per_node=1 main_dino.py --arch vit_small --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/$EXPERIMENT_NAME --batch_size_per_gpu 256 --saveckp_freq 10 --gpu 1 --world_size 1 --dataset CIFAR-100 --epochs 800 --seed $SEED
