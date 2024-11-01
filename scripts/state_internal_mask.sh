#!/bin/bash
export PYTHONPATH=/path/to/your/project
export CUDA_VISIBLE_DEVICES=7
result_path="/path/to/your/output"
#----------------------------------------
exp_name='exp_name_placeholder'
#----------------------------------------
model_name='model_placeholder' # use cache
#----------------------------------------
num_classes="num_placeholder"
#----------------------------------------
cache_layers="23"
#----------------------------------------
cache_types='type_placeholder'
#----------------------------------------
theta=0.3
#----------------------------------------
data_dir=${result_path}'/'${exp_name}'/features/hdata'
save_dir=${result_path}'/'${exp_name}'/masks/hdata/internal'
#----------------------------------------

python core/state_internal_mask.py \
  --model_name ${model_name} \
  --num_classes ${num_classes} \
  --cache_layers ${cache_layers} \
  --cache_types ${cache_types} \
  --theta ${theta} \
  --data_dir ${data_dir} \
  --save_dir ${save_dir}
