#!/bin/bash
export PYTHONPATH=/path/to/your/project
export CUDA_VISIBLE_DEVICES=7
result_path='/path/to/your/output'
#----------------------------------------
exp_name='exp_placeholder'
#----------------------------------------
num_layers=24
#----------------------------------------
num_classes=10
#----------------------------------------
cache_types='type_placeholder'
#----------------------------------------
theta=0.3
#----------------------------------------
data_dir='/path/to/your/data/features/'
save_dir='/path/to/your/data/scores/'
#----------------------------------------

python core/state_internal_score.py \
  --num_layers ${num_layers} \
  --num_classes ${num_classes} \
  --cache_types ${cache_types} \
  --theta ${theta} \
  --data_dir ${data_dir} \
  --save_dir ${save_dir}
