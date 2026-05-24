#!/bin/bash

python -m vmcnet.train.mcmc_test \
  --reload.logdir="./reload_restore" \
  --reload.config_relative_file_path="formamide61687.json" \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path="formamide61687_150000.npz" \
  --test_nwalkers=8 \
  --test_geom_indices=19 \
  --test_H_idx=4 \
  --test_n_macro_steps=50 \
  --test_burn_in_macro=10 \
  --test_thin_macro=1 \
  --test_compute_energy=False \
  --test_energy_every=1 \
  --test_zeta_H=1.0 \
  --test_variants=normal,H_down_seed \
  --test_results_root=../test_results \
  --test_suffix=test