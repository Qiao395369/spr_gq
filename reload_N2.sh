#!/bin/bash

python -m vmcnet.train.runners \
--reload.logdir="./" \
--reload.use_config_file=True \
--reload.config_relative_file_path="reload_restore/config.json" \
--reload.use_checkpoint_file=True \
--reload.checkpoint_relative_file_path="reload_restore/best_checkpoint.npz" \
--reload.new_optimizer_state=False \
--reload.reburn=False \
--reload.append=True \
--reload.same_logdir=False 
