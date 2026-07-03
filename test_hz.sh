#!/bin/bash

python -m vmcnet.train.hz_test \
  --reload.logdir="./reload_restore" \
  --reload.config_relative_file_path="config73297.json" \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path="formamide73297_140000.npz" \