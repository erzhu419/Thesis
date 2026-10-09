#!/bin/bash
set -e
exec /usr/bin/env PATH=/home/erzhu419/anaconda3/envs/LSTM-RL/bin:/usr/bin:/bin LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcurl.so.4 PYTHONPATH=/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/h2oplus_v20_formal_confirmation_evaluation_v4/workspace/H2Oplus LIBSUMO_AS_TRACI=quiet OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/erzhu419/anaconda3/envs/LSTM-RL/bin/python "$@"
