#!/bin/bash

vmc-molecule \
--reload.logdir=../aaa_results/N2_preliminary_ss_3/ \
--reload.checkpoint_relative_file_path=checkpoints/20000.npz \
--reload.new_optimizer_state=True \
--reload.append=False \
--config.logdir=./results \
--config.vmc.nchains=64 \
--config.vmc.nepochs=100 \
--config.eval.nchains=128 \
--config.eval.nepochs=20000 \
--config.vmc.optimizer_type=spring \
--config.vmc.optimizer.spring.learning_rate=0.02