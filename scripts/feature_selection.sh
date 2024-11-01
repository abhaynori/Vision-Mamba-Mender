#!/bin/bash
export PYTHONPATH=/path/to/your/project
export CUDA_VISIBLE_DEVICES=2
result_path="/path/to/your/output"
#----------------------------------------
exp_name='exp_name_placeholder'
#----------------------------------------
model_name='model_placeholder' # use cache
#----------------------------------------
data_name="data_placeholder"
num_classes=50
#----------------------------------------
model_path='/path/to/your/model/file/model_int.pth'
#----------------------------------------
num_samples="num_placeholder"
data_dir='/path/to/your/dataset/train-m/'
save_dir='/path/to/your/output/features_int/'
#----------------------------------------
cache_layers="layer_placeholder"
#----------------------------------------
cache_types=(
"cache_types_placeholder"
)
value_types=(
  'a'
  'g'
)
for value_type in ${value_types[*]}; do
  for cache_type in ${cache_types[*]}; do
    python core/feature_selection.py \
      --model_name ${model_name} \
      --data_name ${data_name} \
      --num_classes ${num_classes} \
      --model_path ${model_path} \
      --num_samples ${num_samples} \
      --cache_layers ${cache_layers} \
      --cache_type "${cache_type}" \
      --value_type "${value_type}" \
      --data_dir ${data_dir} \
      --save_dir ${save_dir}
  done
done
wait
echo "ACCOMPLISH ！！！"
