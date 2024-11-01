#!/bin/bash
export PYTHONPATH=/path/to/your/project
export CUDA_VISIBLE_DEVICES=2
result_path='/path/to/your/output'
#----------------------------------------
exp_name='exp_name_placeholder'
#----------------------------------------
num_layers=24
#----------------------------------------
num_classes="num_placeholder"
#----------------------------------------
cache_types='type_placeholder'
#----------------------------------------
sample_dir='/path/to/your/sample_dir/'
sample_name_path='/path/to/your/sample_name_path.pkl'
data_dir='/path/to/your/data_dir/'
save_dir='/path/to/your/save_dir/'
#----------------------------------------

python core/state_external_mask.py \
  --num_layers ${num_layers} \
  --num_classes ${num_classes} \
  --cache_types ${cache_types} \
  --sample_dir ${sample_dir} \
  --sample_name_path ${sample_name_path} \
  --data_dir ${data_dir} \
  --save_dir ${save_dir}
