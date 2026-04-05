#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/dcase-2022-task4
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dcase2022
export OMP_NUM_THREADS=1

python train_sed.py \
  --conf_file ./confs/crnn_beats_residual_gated_fusion_a800_finetune.yaml \
  --log_dir ./exp/crnn_beats_residual_gated_fusion_a800_finetune \
  --gpus 1 \
  --synth_only
