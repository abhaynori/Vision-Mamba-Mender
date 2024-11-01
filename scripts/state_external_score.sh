#!/bin/bash
export PYTHONPATH=/path/to/your/project
export CUDA_VISIBLE_DEVICES=6
result_path='/path/to/your/output'
#----------------------------------------
exp_name='exp_name_placeholder'
#----------------------------------------
model_name='model_placeholder'
#----------------------------------------
data_name='data_placeholder'
num_classes=10
#---------------------------------------
theta=0.5
##-----------------------------------------
model_path='/path/to/your/model.pth'
#----------------------------------------
data_dir='/path/to/your/dataset/train/'
mask_gt_dir='/path/to/your/mask/'
mask_pd_dir='/path/to/your/masks_pd/{}/layer{}'
save_dir='/path/to/your/scores/'
#----------------------------------------

python core/state_external_score.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --theta ${theta} \
  --model_path ${model_path} \
  --data_dir ${data_dir} \
  --mask_gt_dir ${mask_gt_dir} \
  --mask_pd_dir ${mask_pd_dir} \
  --save_dir ${save_dir}
