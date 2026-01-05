#!/bin/bash

python -m vmcnet.train.runners \
--reload.logdir="./" \
--reload.use_config_file=True \
--reload.config_relative_file_path="reload_restore/config.json" \
--reload.use_checkpoint_file=True \
--reload.checkpoint_relative_file_path="reload_restore/120000.npz" \
--reload.new_optimizer_state=False \
--reload.reburn=False \
--reload.append=True \
--reload.to_pmap=False \
--reload.same_logdir=False \
--reload.new_data=True \
--reload.nchains=44 \
--reload.end_epochs=120100 \
--reload.nburn=100 \
--reload.down_sample_num=4 \
--reload.n_inner=2 \