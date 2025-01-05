#!/bin/sh
#export CUDA_VISIBLE_DEVICES="1"

echo "开始推理"
parallel -j 2 ::: \
  "python -u incontext_generation.py -m llama3 -t ours -d basic_dataset_far2near_error_2_error_middle_our.pkl -n our_dataset_far2near_error_2_error_middle_llama3_our_method  > log/our_dataset_far2near_error_2_error_middle_llama3_our_method.log 2>&1" \
  "python -u incontext_generation.py -m llama3 -t ours -d basic_dataset_far2near_error_6_error_middle_our.pkl -n our_dataset_far2near_error_6_error_middle_llama3_our_method  > log/our_dataset_far2near_error_6_error_middle_llama3_our_method.log 2>&1" \
  "python -u incontext_generation.py -m llama3 -t ours -d basic_dataset_near2far_error_2_error_middle_our.pkl -n our_dataset_near2far_error_2_error_middle_llama3_our_method  > log/our_dataset_near2far_error_2_error_middle_llama3_our_method.log 2>&1" \
  "python -u incontext_generation.py -m llama3 -t ours -d basic_dataset_near2far_error_6_error_middle_our.pkl -n our_dataset_near2far_error_6_error_middle_llama3_our_method  > log/our_dataset_near2far_error_6_error_middle_llama3_our_method.log 2>&1" \
  "python -u incontext_generation.py -m llama3 -t ours -d basic_dataset_suddenly_appeal_error_middle_our.pkl  -n our_dataset_suddenly_appeal_error_middle_llama3_our_method   > log/our_dataset_suddenly_appeal_error_middle_llama3_gour_method.log 2>&1"

#echo "完成无异常+远到近异常"
#
#
#parallel ::: \

#


echo "推理完成"
rm /data/code/xuzheling/llm_autodriver_error_handle/outputs/pkl/*text*
rm /data/code/xuzheling/llm_autodriver_error_handle/outputs/pkl/*jsonl

echo "删除无用输出"


echo "评估"
