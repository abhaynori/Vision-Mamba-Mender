#!/bin/bash
export PYTHONPATH=/path/to/your/project
export CUDA_VISIBLE_DEVICES=0
result_path="/path/to/your/outputs"
#----------------------------------------
model_name='model_placeholder'
#----------------------------------------
num_classes="num_placeholder"
data_name="data_placeholder"
#----------------------------------------
num_epochs="num_placeholder"
batch_size=256
lr=0.001
#----------------------------------------
data_train_dir="/path/to/your/dataset/train"
data_test_dir="/path/to/your/dataset/test"
#----------------------------------------
exp_name="${model_name}_${data_name}_bs${batch_size}_lr${lr}_cos_wm_new"
#----------------------------------------
model_dir="${result_path}/${exp_name}/models"
#----------------------------------------
log_dir=${result_path}'/runs/'${exp_name}

python engines/train.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --num_epochs ${num_epochs} \
  --batch-size ${batch_size} \
  --lr ${lr} \
  --model_dir ${model_dir} \
  --data_train_dir ${data_train_dir} \
  --data_test_dir ${data_test_dir} \
  --log_dir ${log_dir} \
