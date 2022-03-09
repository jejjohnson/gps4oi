#!/bin/bash

oarsub \
    -l /nodes=1/gpu=1,walltime=01:00:00 \
    -p "gpumodel='V100'" \
    --project pr-data-ocean \
    --stdout /bettik/johnsonj/logs/gps4oi_opt.log \
    --stderr /bettik/johnsonj/errs/gps4oi_opt.log \


conda activate oi_torch_gpu_py39
python -u /bettik/johnsonj/projects/gps4oi/scripts/baselines/opt.py