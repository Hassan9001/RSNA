#!/bin/bash
#SBATCH --job-name=TOFMRA_ResEnc_45kVoCo_lr1e-4_5folds_bsize8
#SBATCH --time=24:00:00
#SBATCH -c 20
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --output=logs/TOFMRA_ResEnc_45kVoCo_lr1e-4_5folds_bsize8_%A_%a.out
#SBATCH --error=logs/TOFMRA_ResEnc_45kVoCo_lr1e-4_5folds_bsize8_%A_%a.err

# Load micromamba shell function
eval "$(/hpf/projects/jquon/micromamba/bin/micromamba shell hook -s bash)"

# Activate your environment
micromamba activate ssl3d

#TODO: change path for EXPERIMENT_LOCATION
export EXPERIMENT_LOCATION=/hpf/projects/jquon/sumin/nnssl_dataset/nnssl_results/Dataset004_AVM_TOFMRA/ResEnc_45kVoCo_pretrained_cls_DblWarmup_lr_1e-4_5folds_TOFMRA_bsize8
export WANDB_DIR=$EXPERIMENT_LOCATION/wandb
export WANDB_CACHE_DIR=$EXPERIMENT_LOCATION/wandb_cache
export TORCH_HOME=$EXPERIMENT_LOCATION/torch_cache
export TMPDIR=/hpf/projects/jquon/tmp/tmp
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$TORCH_HOME"

cd /hpf/projects/jquon/models/SSL3D_classification

#TODO: change data, trainer.logger.name
python main_5folds.py env=cluster model=resenc data=AVM_TOFMRA_data \
  trainer.devices=1 model.pretrained=True \
  data.module.fold=$SLURM_ARRAY_TASK_ID \
  trainer.logger.name=Fold${SLURM_ARRAY_TASK_ID}_ResEnc_45kVoCo_pretrained_cls_DblWarmup_lr_1e-4_5folds_TOFMRA_bsize8 \
  model.chpt_path=/hpf/projects/jquon/sumin/nnssl_checkpoints/checkpoint_final_45kVoCo.pth
