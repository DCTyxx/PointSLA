#!/usr/bin/env bash
set -e
cd /home/xxyu/project/GFS-VL
export PATH=/usr/local/cuda-11.8/bin:/home/xxyu/miniforge3/envs/gfsvl/bin:/usr/bin:/bin
export CUDA_HOME=/usr/local/cuda-11.8
export PYTHONPATH=/home/xxyu/project/GFS-VL
export CUDA_VISIBLE_DEVICES=0
/home/xxyu/miniforge3/envs/gfsvl/bin/python tools/train.py \
  --config-file configs/scannet/semseg-pt-v3m1-0-gfsregistrain_k1.py \
  --num-gpus 1 \
  --options \
  save_path=exp/scannet/gfsvl_smoke \
  regis_train_list=[regis1] \
  epoch=1 \
  eval_epoch=1 \
  batch_size=2 \
  num_worker=2
