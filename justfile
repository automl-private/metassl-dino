# List available just commands
@list:
  just --list

# ---------------------------------------------------------------------------------------
# CIFAR-10 (DIANE)
# ---------------------------------------------------------------------------------------
# Run DINO pretraining for CIFAR-10 locally (Diane)
@cifar10_pt_local EXPERIMENT_NAME EPOCHS:
  #!/usr/bin/env bash
  python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_small --output_dir experiments/{{EXPERIMENT_NAME}} --batch_size_per_gpu 64 --saveckp_freq 10 --gpu 1 --world_size 1 --dataset CIFAR-10 --epochs {{EPOCHS}}

# Run DINO pretraining for CIFAR-10 on the cluster (Diane)
@cifar10_pt EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu16,dlcgpu17 --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_cifar10_dino_pretraining.sh

# Run DINO NEPS (data augmentation) for CIFAR-10 locally (Diane)
@cifar10_neps_data_augmentation_local EXPERIMENT_NAME EPOCHS:
  #!/usr/bin/env bash
  mkdir -p /tmp/dino_communication
  filename=/tmp/dino_communication/$(openssl rand -hex 12)
  python -u -m torch.distributed.launch --use_env --nproc_per_node=1 --nnodes=1 main_dino.py --config_file_path $filename --arch vit_small --output_dir experiments/{{EXPERIMENT_NAME}} --batch_size_per_gpu 64 --saveckp_freq 10 --gpu 1 --world_size 1 --dataset CIFAR-10 --epochs {{EPOCHS}} --is_neps_run
  rm filename

# Run DINO NEPS (data augmentation) for CIFAR-10 on the cluster (Diane)
@cifar10_neps_data_augmentation EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu16,dlcgpu17 --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_cifar10_dino_neps_data_augmentation.sh

# Run DINO NEPS (groupaugment) for CIFAR-10 on the cluster (Diane)
@cifar10_neps_groupaugment EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu46 --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_cifar10_dino_neps_groupaugment.sh

# Run DINO NEPS (training) for CIFAR-10 locally (Diane)
@cifar10_neps_training_local EXPERIMENT_NAME EPOCHS:
  #!/usr/bin/env bash
  mkdir -p /tmp/dino_communication
  filename=/tmp/dino_communication/$(openssl rand -hex 12)
  python -u -m torch.distributed.launch --use_env --nproc_per_node=1 --nnodes=1 main_dino.py --config_file_path $filename --arch vit_small --output_dir experiments/{{EXPERIMENT_NAME}} --batch_size_per_gpu 64 --saveckp_freq 10 --gpu 1 --world_size 1 --dataset CIFAR-10 --epochs {{EPOCHS}} --is_neps_run --config_space training
  rm filename

# Run DINO NEPS (training) for CIFAR-10 on the cluster (Diane)
@cifar10_neps_training EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu16,dlcgpu17 --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_cifar10_dino_neps_training.sh

# Run linear DINO evaluation for CIFAR-10 locally (Diane)
@cifar10_eval_local EXPERIMENT_NAME EPOCHS:
  #!/usr/bin/env bash
  python -m torch.distributed.launch --nproc_per_node=1 eval_linear.py --pretrained_weights experiments/{{EXPERIMENT_NAME}}/checkpoint.pth --arch vit_small --output_dir experiments/{{EXPERIMENT_NAME}} --batch_size_per_gpu 64 --gpu 1 --world_size 1 --dataset CIFAR-10 --epochs {{EPOCHS}}

# Run linear DINO evaluation for CIFAR-10 on the cluster (Diane)
@cifar10_eval EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu05,dlcgpu29,dlcgpu09 --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_cifar10_dino_linear_evaluation.sh

# ---------------------------------------------------------------------------------------
# CIFAR-100 (DIANE)
# ---------------------------------------------------------------------------------------
# Run DINO pretraining for CIFAR-100 locally (Diane)
@cifar100_pt_local EXPERIMENT_NAME EPOCHS:
  #!/usr/bin/env bash
  python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_small --output_dir experiments/{{EXPERIMENT_NAME}} --batch_size_per_gpu 64 --saveckp_freq 10 --gpu 1 --world_size 1 --dataset CIFAR-100 --epochs {{EPOCHS}}

# Run DINO pretraining for CIFAR-100 on the cluster (Diane)
@cifar100_pt EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu17,dlcgpu29,dlcgpu09,dlcgpu14 --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_cifar100_dino_pretraining.sh

# Run DINO NEPS (data augmentation) for CIFAR-100 locally (Diane)
@cifar100_neps_data_augmentation_local EXPERIMENT_NAME EPOCHS:
  #!/usr/bin/env bash
  mkdir -p /tmp/dino_communication
  filename=/tmp/dino_communication/$(openssl rand -hex 12)
  python -u -m torch.distributed.launch --use_env --nproc_per_node=1 --nnodes=1 main_dino.py --config_file_path $filename --arch vit_small --output_dir experiments/{{EXPERIMENT_NAME}} --batch_size_per_gpu 64 --saveckp_freq 10 --gpu 1 --world_size 1 --dataset CIFAR-100 --epochs {{EPOCHS}} --is_neps_run
  rm filename

# Run DINO NEPS (data augmentation) for CIFAR-100 on the cluster (Diane)
@cifar100_neps_data_augmentation EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu16,dlcgpu17 --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_cifar100_dino_neps_data_augmentation.sh

# Run DINO NEPS (groupaugment) for CIFAR-100 on the cluster (Diane)
@cifar100_neps_groupaugment EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu46 --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_cifar100_dino_neps_groupaugment.sh

# Run DINO NEPS (training) for CIFAR-100 locally (Diane)
@cifar100_neps_training_local EXPERIMENT_NAME EPOCHS:
  #!/usr/bin/env bash
  mkdir -p /tmp/dino_communication
  filename=/tmp/dino_communication/$(openssl rand -hex 12)
  python -u -m torch.distributed.launch --use_env --nproc_per_node=1 --nnodes=1 main_dino.py --config_file_path $filename --arch vit_small --output_dir experiments/{{EXPERIMENT_NAME}} --batch_size_per_gpu 64 --saveckp_freq 10 --gpu 1 --world_size 1 --dataset CIFAR-100 --epochs {{EPOCHS}} --is_neps_run --config_space training
  rm filename

# Run DINO NEPS (training) for CIFAR-100 on the cluster (Diane)
@cifar100_neps_training EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu16,dlcgpu17 --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_cifar100_dino_neps_training.sh

# Run linear DINO evaluation for CIFAR-100 locally (Diane)
@cifar100_eval_local EXPERIMENT_NAME EPOCHS:
  #!/usr/bin/env bash
  python -m torch.distributed.launch --nproc_per_node=1 eval_linear.py --pretrained_weights experiments/{{EXPERIMENT_NAME}}/checkpoint.pth --arch vit_small --output_dir experiments/{{EXPERIMENT_NAME}} --batch_size_per_gpu 64 --gpu 1 --world_size 1 --dataset CIFAR-100 --epochs {{EPOCHS}}

# Run linear DINO evaluation for CIFAR-100 on the cluster (Diane)
@cifar100_eval EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/CIFAR-100/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_cifar100_dino_linear_evaluation.sh

# ---------------------------------------------------------------------------------------
# ImageNet (DIANE + DANNY)
# ---------------------------------------------------------------------------------------

# Run DINO pretraining for ImageNet on the cluster (Diane)
@imagenet_pt EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_imagenet_dino_pretraining.sh

# Run linear DINO evaluation for ImageNet on the cluster (Diane)
@imagenet_eval EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu24 --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_imagenet_dino_linear_evaluation.sh

# Run DINO pretraining for ImageNet on the cluster (Danny)
@imagenet_pt_best_config EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/stolld-metassl_dino/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge1/stolld-metassl_dino/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge1/stolld-metassl_dino/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_imagenet_dino_pretraining_best_config.sh

# Run linear DINO evaluation for ImageNet on the cluster (Danny)
@imagenet_eval_best_config EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/stolld-metassl_dino/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge1/stolld-metassl_dino/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge1/stolld-metassl_dino/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_imagenet_dino_linear_evaluation_best_config.sh

# Run DINO pretraining for ImageNet on the cluster with ResNet50 (Diane)
@imagenet_pt_resnet50 EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_imagenet_dino_pretraining_resnet50.sh

# Run linear DINO evaluation for ImageNet on the cluster with ResNet50 (Diane)
@imagenet_eval_resnet50 EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_imagenet_dino_linear_evaluation_resnet50.sh

# Run DINO NEPS (data augmentation) for ImageNet on the cluster (Diane)
@imagenet_neps_data_augmentation EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu47 --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_imagenet_dino_neps_data_augmentation.sh

# Run DINO NEPS (training) for ImageNet on the cluster (Danny)
@imagenet_neps_training EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/stolld-metassl_dino/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu47 --output=/work/dlclarge1/stolld-metassl_dino/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge1/stolld-metassl_dino/dino/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_imagenet_dino_neps_training.sh

# ---------------------------------------------------------------------------------------
# Transfer to other datasets (Danny)
# ---------------------------------------------------------------------------------------
# Transfer ImageNet pre-trained weigths (from baseline) to other datasets
@transfer_DA_weights DATASET EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/stolld-metassl_dino/dino/{{DATASET}}/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge1/stolld-metassl_dino/dino/{{DATASET}}/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge1/stolld-metassl_dino/dino/{{DATASET}}/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}},DATASET={{DATASET}} cluster/submit_transfer_to_dataset.sh

# ---------------------------------------------------------------------------------------
# Transfer to other datasets (DIANE)
# ---------------------------------------------------------------------------------------
# Transfer ImageNet pre-trained weigths (from baseline) to other datasets
@transfer_baseline_weights DATASET EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/{{DATASET}}/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{DATASET}}/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{DATASET}}/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}},DATASET={{DATASET}} cluster/submit_transfer_to_dataset.sh

# ---------------------------------------------------------------------------------------
# ImageNet
# ---------------------------------------------------------------------------------------

# Run pretraining DINO
@dino_pt EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_dino_pretraining.sh

# Test pretraining DINO
@test_dino_pt EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_test_dino_pretraining.sh

# Run finetuning DINO
@dino_ft EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_dino_finetuning.sh

# Test finetuning DINO
@test_dino_ft EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_test_dino_finetuning.sh

# Run NEPS DINO
@dino_neps EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_dino_neps.sh

# Test NEPS DINO
@test_dino_neps EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_test_dino_neps.sh

# Run NEPS DINO Fabio
@dino_neps_fabio EXPERIMENT_NAME:
  #!/usr/bin/env zsh
  mkdir -p /work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_dino_neps_fabio.sh

# Test NEPS DINO Fabio
@dino_neps_fabio_test EXPERIMENT_NAME:
  #!/usr/bin/env zsh
  mkdir -p /work/dlclarge2/ferreira-dino/metassl-dino/experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/dino/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_dino_neps_fabio_test.sh

# Test without NEPS DINO Fabio (Distributed fix)
@dino_wo_neps_fabio_test EXPERIMENT_NAME:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
   sbatch --output=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_dino_wo_neps_fabio.sh

# Run NEPS DINO Fabio Sam (Distributed fix)
@dino_neps_distributed_fix EXPERIMENT_NAME:
  #!/usr/bin/env zsh
  mkdir -p /work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_dino_neps_fabio_sam.sh

@dino_neps_imagenet_balanced_val_data_augmentation_1 EXPERIMENT_NAME:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
   sbatch --output=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_dino_neps_fabio_data_augmentation.sh

@dino_neps_imagenet_balanced_val_training_hypers EXPERIMENT_NAME:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
   sbatch --output=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_imagenet_dino_neps_fabio_train_hypers.sh

@dino_neps_imagenet_balanced_val_data_augmentation_2 EXPERIMENT_NAME:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
   sbatch --output=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_dino_neps_fabio_data_augmentation.sh


@imagenet_eval_fabio EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_imagenet_dino_linear_evaluation.sh

@imagenet_eval_fixed_hypers_fabio EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/ferreira-dino/metassl-dino-merged/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/ferreira-dino/metassl-dino-merged/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino/metassl-dino-merged/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_imagenet_eval_fixed_hypers.sh

@imagenet_short_pretraining_fabio EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_dino_short_pretraining.sh
