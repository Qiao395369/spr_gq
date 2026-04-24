#!/bin/bash

# 正确赋值：不能有空格！
JOB_DIR="../local_energy/"

ENERGY_FILE="${JOB_DIR}multi_energy54852.txt"
OUTPUT_FILE="${JOB_DIR}statistics548526666"

python -m vmcnet.train.do_statistic \
  --local_energies_file_path "$ENERGY_FILE" \
  --output_file_path "$OUTPUT_FILE" \
  --walkers 1 \
  --nchains 512 \
  --cut 4000