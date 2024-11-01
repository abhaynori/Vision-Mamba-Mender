#!/bin/bash
export PYTHONPATH=/path/to/your/project
export CUDA_VISIBLE_DEVICES=1
result_path='/path/to/your/output'
#----------------------------------------
exp_name='exp_name_placeholder'
#----------------------------------------
model_name='model_placeholder'
#----------------------------------------
num_classes=50
#----------------------------------------
model_path=${result_path}'/'${exp_name}'/models/model_ori.pth'
#----------------------------------------
image_path='/path/to/your/image.jpeg'
#----------------------------------------
save_dir=${result_path}'/'${exp_name}'/cams'

python core/mamba_cam.py \
  --model_name ${model_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --image_path ${image_path} \
  --save_dir ${save_dir}
