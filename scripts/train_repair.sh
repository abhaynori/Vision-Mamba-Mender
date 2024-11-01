#!/bin/bash
export PYTHONPATH=/path/to/your/project
export CUDA_VISIBLE_DEVICES=0
result_path="/path/to/your/output"
#----------------------------------------
base_exp_name="exp_name_placeholder"
#----------------------------------------
model_name="model_placeholder"
#----------------------------------------
data_name="data_placeholder"
num_classes="num_placeholder"
#----------------------------------------
data_train_dir="/path/to/your/dataset/train"
data_test_dir="/path/to/your/dataset/test"
#----------------------------------------
batch_size=256
num_epochs="num_placeholder"
lr=0.001
#----------------------------------------
model_path="${result_path}/${base_exp_name}/models/model_ori.pth"
#################### external #####################
#----------------------------------------
external_cache_layers="23"
#----------------------------------------
external_cache_types="c s"
#----------------------------------------
external_mask_dir="/path/to/your/external/mask"
#----------------------------------------
alpha=1e+7
#----------------------------------------
#################### external #####################
#################### internal #####################
#----------------------------------------
internal_cache_layers=""
#internal_cache_layers="23"
#----------------------------------------
internal_cache_types="x"
#----------------------------------------
internal_mask_dir="/path/to/your/internal/mask"
#----------------------------------------
beta=1e+8
#----------------------------------------
#################### internal #####################
#----------------------------------------
exp_name="${base_exp_name}_ext_grad_lay${external_cache_layers}_alp${alpha}_sca"
#----------------------------------------
save_dir="${result_path}/${exp_name}/models"
#----------------------------------------
log_dir="${result_path}/runs/${exp_name}"
#----------------------------------------

python engines/train_repair.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --data_train_dir ${data_train_dir} \
  --data_test_dir ${data_test_dir} \
  --batch-size ${batch_size} \
  --lr ${lr} \
  --num_epochs ${num_epochs} \
  --model_path ${model_path} \
  --alpha ${alpha} \
  --beta ${beta} \
  --external_cache_layers ${external_cache_layers} \
  --internal_cache_layers ${internal_cache_layers} \
  --external_cache_types ${external_cache_types} \
  --internal_cache_types ${internal_cache_types} \
  --external_mask_dir ${external_mask_dir} \
  --internal_mask_dir ${internal_mask_dir} \
  --save_dir ${save_dir} \
  --log_dir ${log_dir}
