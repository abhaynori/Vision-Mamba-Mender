#!/bin/bash
export PYTHONPATH=/path/to/your/project
export CUDA_VISIBLE_DEVICES=0
result_path="/path/to/your/output"
#----------------------------------------
exp_name='exp_name_placeholder'
#----------------------------------------
model_name='model_placeholder'
#----------------------------------------
data_name="data_placeholder"
num_classes=""
#----------------------------------------
model_path=${result_path}'/'${exp_name}'/models/model_ori.pth'
#----------------------------------------
data_dir="/path/to/your/dataset/train"
#----------------------------------------
save_dir=${result_path}'/'${exp_name}'/samples/htrain'
#----------------------------------------
num_samples=""
#----------------------------------------

python core/sample_selection.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_dir ${data_dir} \
  --save_dir ${save_dir} \
  --num_samples ${num_samples} \
  --is_high_confidence
