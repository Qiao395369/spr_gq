#!/bin/bash

python -m vmcnet.train.do_statistic \
  --id "54852" \
  --walkers 1 \
  --nchains 512 \
  --cut 2000
