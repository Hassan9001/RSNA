#!/bin/bash
#SBATCH --job-name=T1C_x2_NewHead_balanced_ResEnc_45kVoCo_pretrained_cls_DblWarmup_lr_1e-4_5folds_bsize8
#SBATCH --time=24:00:00
#SBATCH -c 20
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --array=0-4
#SBATCH --output=logs/T1C_x2_NewHead_balanced_ResEnc_45kVoCo_pretrained_cls_DblWarmup_lr_1e-4_5folds_bsize8_%A_%a.out
#SBATCH --error=logs/T1C_x2_NewHead_balanced_ResEnc_45kVoCo_pretrained_cls_DblWarmup_lr_1e-4_5folds_bsize8_%A_%a.err

# Load micromamba shell function
eval "$(/hpf/projects/jquon/micromamba/bin/micromamba shell hook -s bash)"

# Activate your environment
micromamba activate ssl3d

#TODO: change path for EXPERIMENT_LOCATION
export EXPERIMENT_LOCATION=/hpf/projects/jquon/sumin/nnssl_dataset/nnssl_results/Dataset003_AVM_T1+C/x2_NewHead_balanced_ResEnc_45kVoCo_pretrained_cls_DblWarmup_lr_1e-4_5folds_bsize8
export WANDB_DIR=$EXPERIMENT_LOCATION/wandb
export WANDB_CACHE_DIR=$EXPERIMENT_LOCATION/wandb_cache
export TORCH_HOME=$EXPERIMENT_LOCATION/torch_cache
export TMPDIR=/hpf/projects/jquon/tmp/tmp
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$TORCH_HOME"

cd /hpf/projects/jquon/models/SSL3D_classification

#TODO: change data(i.e., points to AVM_T1+C_data_v2.yaml in ./cli_configs/data/ ??) and experiment name
python main_5folds.py env=cluster model=resenc data=AVM_T1+C_data_NewHead_lr \
  trainer.devices=1 model.pretrained=True \
  data.module.fold=$SLURM_ARRAY_TASK_ID \
  trainer.logger.name=x2_NewHead_balanced_Fold${SLURM_ARRAY_TASK_ID}_ResEnc_45kVoCo_pretrained_cls_DblWarmup_lr_1e-4_5folds_bsize8 \
  model.chpt_path=/hpf/projects/jquon/sumin/nnssl_checkpoints/checkpoint_final_45kVoCo.pth
  # data.module.batch_size=8 \
  # model.chpt_path=/hpf/projects/jquon/sumin/nnssl_checkpoints/checkpoint_final_45kVoCo.pth