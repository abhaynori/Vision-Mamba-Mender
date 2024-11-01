#!/bin/bash
export PYTHONPATH=/path/to/your/project
export CUDA_VISIBLE_DEVICES=1
#result_path='/path/to/your/output'
#----------------------------------------
#exp_name='exp_name_placeholder' # model_placeholder
#----------------------------------------
#feature_dir=${result_path}'/'${exp_name}'/features/data'
#save_dir=${result_path}'/'${exp_name}'/features'
#----------------------------------------

python core/feature_visualize.py \
#  --feature_dir ${feature_dir} \
#  --save_dir ${save_dir}
