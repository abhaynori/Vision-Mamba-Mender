#!/bin/bash
export PYTHONPATH=/path/to/your/project
export CUDA_VISIBLE_DEVICES=1
result_path='/path/to/your/output'
#----------------------------------------
exp_name='exp_placeholder'
#----------------------------------------
model_name='model_placeholder'
#----------------------------------------
data_name='data_placeholder'
num_classes=50
#----------------------------------------
model_path=${result_path}'/'${exp_name}'/models/model_ori.pth'
#----------------------------------------
#data_dir=${result_path}'/'${exp_name}'/images/htest'
data_dir=${result_path}'/'${exp_name}'/images/htrain'
#----------------------------------------
save_dir=${result_path}'/'${exp_name}'/mechs/htest_spatial_channel'
#----------------------------------------

python core/mamba_mech.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_dir ${data_dir} \
  --save_dir ${save_dir}
