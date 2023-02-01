#!/usr/bin/env bash

# NOTE: This script requires you to have already installed the `didactic` and `vital` packages in your python
# environment

RESULTS_DIR=$1

for model_dir in ${RESULTS_DIR}/*; do # Iterate over each segmentation method
  model_dir=${model_dir%*/} # remove the trailing "/"
  model_basename="$(basename "${model_dir}")"

  echo "=============== $model_basename ==============="

  python vital/vital/results/cardinal/temporal_metrics.py \
    --data_roots "$model_dir" --views A4C A2C \
    --output_folder="$model_dir" \
    --input=post_pred_mask

done
