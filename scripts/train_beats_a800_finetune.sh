#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/dcase-2022-task4
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dcase2022
export OMP_NUM_THREADS=1

python train_sed.py \
  --conf_file ./confs/unified_beats_synth_only_a800_finetune.yaml \
  --log_dir ./exp/unified_beats_synth_only_a800_finetune \
  --gpus 1 \
  --synth_only
