#!/bin/bash
export PYTHONPATH=/path/to/your/project
export CUDA_VISIBLE_DEVICES=7
result_path="/path/to/your/output"
#----------------------------------------
exp_name='exp_name_placeholder'
#----------------------------------------
model_name='model_placeholder'
#----------------------------------------
data_name="data_placeholder"
num_classes="num_placeholder"
#----------------------------------------
data_dir="/path/to/your/dataset/test"
#----------------------------------------
model_path="${result_path}/${exp_name}/models/model_name.pth"
#----------------------------------------

python engines/test.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_dir ${data_dir}
