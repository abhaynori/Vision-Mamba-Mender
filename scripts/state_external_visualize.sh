#!/bin/bash
export PYTHONPATH=/path/to/your/project
export CUDA_VISIBLE_DEVICES=0
result_path='/path/to/your/project/outputs'
#----------------------------------------
exp_name='exp_placeholder'
#----------------------------------------
num_layers="num_placeholder"
#----------------------------------------
num_classes="num_placeholder"
#----------------------------------------
num_samples="num_placeholder"
#----------------------------------------
cache_types="cache_types"
#----------------------------------------
sample_dir=${result_path}'/'${exp_name}'/samples/htrain'
sample_name_path=${result_path}'/'${exp_name}'/features/hdata/sample_names.pkl'
data_dir=${result_path}'/'${exp_name}'/features/hdata/'
save_dir=${result_path}'/'${exp_name}'/visualize/hdata/external'
#----------------------------------------

python core/state_external_visualize.py \
  --num_layers ${num_layers} \
  --num_classes ${num_classes} \
  --num_samples ${num_samples} \
  --cache_types ${cache_types} \
  --sample_dir ${sample_dir} \
  --sample_name_path ${sample_name_path} \
  --data_dir ${data_dir} \
  --save_dir ${save_dir}
