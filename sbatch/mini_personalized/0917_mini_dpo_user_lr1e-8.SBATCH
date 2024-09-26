#!/bin/bash

#SBATCH --constraint='a100|h100'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=0917_mini_dpo_user_lr1e-8

module purge

singularity exec --nv \
    --overlay /scratch/yl11330/my_env/overlay-50G-10M-pytorch.ext3:ro \
    --overlay /scratch/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; cd /scratch/yl11330/personalized_diffusion; conda activate ./penv; \
        ./launchers/mini_personalized/mini_dpo_user_lr1e-8.sh"

# 51078287 (me)