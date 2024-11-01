#!/bin/bash
export PYTHONPATH=/path/to/your/project/
export CUDA_VISIBLE_DEVICES=7
result_path="/path/to/your/outputs"
#----------------------------------------
exp_name='exp_name_placeholder'
#----------------------------------------
num_classes=50
#----------------------------------------
cache_layers="layer_placeholder"
#----------------------------------------
cache_types='type_placeholder'
#----------------------------------------
sample_dir=${result_path}'/'${exp_name}'/samples/htrain'
data_dir=${result_path}'/'${exp_name}'/features/hdata'
save_dir=${result_path}'/'${exp_name}'/visualize/hdata/internal'
#----------------------------------------
#sample_dir=${result_path}'/'${exp_name}'/samples/ltrain'
#data_dir=${result_path}'/'${exp_name}'/features/ldata'
#save_dir=${result_path}'/'${exp_name}'/visualize/ldata/internal'
#----------------------------------------

python core/state_internal_visualize.py \
  --num_classes ${num_classes} \
  --cache_layers ${cache_layers} \
  --cache_types ${cache_types} \
  --sample_dir ${sample_dir} \
  --data_dir ${data_dir} \
  --save_dir ${save_dir}
