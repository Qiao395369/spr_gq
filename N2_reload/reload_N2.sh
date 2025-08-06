#!/bin/bash

vmc-molecule \
--reload.logdir="./" \
--reload.use_config_file=True \
--reload.config_relative_file_path="config.json" \
--reload.use_checkpoint_file=True \
--reload.checkpoint_relative_file_path="checkpoints/150000.npz" \
--reload.new_optimizer_state=False \
--reload.reburn=False \
--reload.append=True \
--reload.same_logdir=False 
