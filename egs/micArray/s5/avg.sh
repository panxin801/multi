#!/bin/bash
source path.sh
expdir=exp/multiASR

python $MAIN_ROOT/src/avg_last_ckpts.py \
    $expdir \
    10


