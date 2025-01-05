#!/bin/sh

echo "开始推理"
#parallel -j 2 ::: \
#  "python -u incontext_generation.py -m llama3 -t gpt-driver -d only_collision_after_error/far2near_error_2_error_middle_our_only_collision_after_error.pkl -n far2near_error_2_error_middle_our_only_collision_after_error_llama3_gptdriver  > log/far2near_error_2_error_middle_our_only_collision_after_error_llama3_gptdriver.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t ours -d only_collision_after_error/far2near_error_2_error_middle_our_only_collision_after_error.pkl       -n far2near_error_2_error_middle_our_only_collision_after_error_llama3_our_method > log/far2near_error_2_error_middle_our_only_collision_after_error_llama3_our_method.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t gpt-driver -d only_collision_after_error/far2near_error_6_error_middle_our_only_collision_after_error.pkl -n far2near_error_6_error_middle_our_only_collision_after_error_llama3_gptdriver  > log/far2near_error_6_error_middle_our_only_collision_after_error_llama3_gptdriver.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t ours -d only_collision_after_error/far2near_error_6_error_middle_our_only_collision_after_error.pkl       -n far2near_error_6_error_middle_our_only_collision_after_error_llama3_our_method > log/far2near_error_6_error_middle_our_only_collision_after_error_llama3_our_method.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t gpt-driver -d only_collision_after_error/near2far_error_2_error_middle_our_only_collision_after_error.pkl -n near2far_error_2_error_middle_our_only_collision_after_error_llama3_gptdriver  > log/near2far_error_2_error_middle_our_only_collision_after_error_llama3_gptdriver.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t ours -d only_collision_after_error/near2far_error_2_error_middle_our_only_collision_after_error.pkl       -n near2far_error_2_error_middle_our_only_collision_after_error_llama3_our_method > log/near2far_error_2_error_middle_our_only_collision_after_error_llama3_our_method.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t gpt-driver -d only_collision_after_error/near2far_error_6_error_middle_our_only_collision_after_error.pkl -n near2far_error_6_error_middle_our_only_collision_after_error_llama3_gptdriver  > log/near2far_error_6_error_middle_our_only_collision_after_error_llama3_gptdriver.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t ours -d only_collision_after_error/near2far_error_6_error_middle_our_only_collision_after_error.pkl       -n near2far_error_6_error_middle_our_only_collision_after_error_llama3_our_method > log/near2far_error_6_error_middle_our_only_collision_after_error_llama3_our_method.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t gpt-driver -d only_collision_after_error/suddenly_appeal_error_middle_our_only_collision_after_error.pkl  -n suddenly_appeal_error_middle_our_only_collision_after_error_llama3_gptdriver  > log/suddenly_appeal_error_middle_our_only_collision_after_error_llama3_gptdriver.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t ours -d only_collision_after_error/suddenly_appeal_error_middle_our_only_collision_after_error.pkl        -n suddenly_appeal_error_middle_our_only_collision_after_error_llama3_our_method > log/suddenly_appeal_error_middle_our_only_collision_after_error_llama3_our_method.log 2>&1" \
#echo "完成只有加了异常后碰撞的数据集"
#
#parallel -j 2 ::: \
#  "python -u incontext_generation.py -m llama3 -t gpt-driver -d without_collision_all/far2near_error_2_error_middle_our_without_collision_all.pkl -n far2near_error_2_error_middle_our_without_collision_all_llama3_gptdriver  > log/far2near_error_2_error_middle_our_without_collision_all_llama3_gptdriver.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t ours -d without_collision_all/far2near_error_2_error_middle_our_without_collision_all.pkl       -n far2near_error_2_error_middle_our_without_collision_all_llama3_our_method > log/far2near_error_2_error_middle_our_without_collision_all_llama3_our_method.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t gpt-driver -d without_collision_all/far2near_error_6_error_middle_our_without_collision_all.pkl -n far2near_error_6_error_middle_our_without_collision_all_llama3_gptdriver  > log/far2near_error_6_error_middle_our_without_collision_all_llama3_gptdriver.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t ours -d without_collision_all/far2near_error_6_error_middle_our_without_collision_all.pkl       -n far2near_error_6_error_middle_our_without_collision_all_llama3_our_method > log/far2near_error_6_error_middle_our_without_collision_all_llama3_our_method.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t gpt-driver -d without_collision_all/near2far_error_2_error_middle_our_without_collision_all.pkl -n near2far_error_2_error_middle_our_without_collision_all_llama3_gptdriver  > log/near2far_error_2_error_middle_our_without_collision_all_llama3_gptdriver.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t ours -d without_collision_all/near2far_error_2_error_middle_our_without_collision_all.pkl       -n near2far_error_2_error_middle_our_without_collision_all_llama3_our_method > log/near2far_error_2_error_middle_our_without_collision_all_llama3_our_method.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t gpt-driver -d without_collision_all/near2far_error_6_error_middle_our_without_collision_all.pkl -n near2far_error_6_error_middle_our_without_collision_all_llama3_gptdriver  > log/near2far_error_6_error_middle_our_without_collision_all_llama3_gptdriver.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t ours -d without_collision_all/near2far_error_6_error_middle_our_without_collision_all.pkl       -n near2far_error_6_error_middle_our_without_collision_all_llama3_our_method > log/near2far_error_6_error_middle_our_without_collision_all_llama3_our_method.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t gpt-driver -d without_collision_all/suddenly_appeal_error_middle_our_without_collision_all.pkl  -n suddenly_appeal_error_middle_our_without_collision_all_llama3_gptdriver  > log/suddenly_appeal_error_middle_our_without_collision_all_llama3_gptdriver.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t ours -d without_collision_all/suddenly_appeal_error_middle_our_without_collision_all.pkl        -n suddenly_appeal_error_middle_our_without_collision_all_llama3_our_method > log/suddenly_appeal_error_middle_our_without_collision_all_llama3_our_method.log 2>&1" \
#echo "完成加异常前没有碰撞加异常侯碰撞+加前加后都没有碰撞的数据集"
#
#parallel -j 2 ::: \
#  "python -u incontext_generation.py -m llama3 -t gpt-driver -d with_collision_all/far2near_error_2_error_middle_our_with_collision_all.pkl -n far2near_error_2_error_middle_our_with_collision_all_llama3_gptdriver  > log/far2near_error_2_error_middle_our_with_collision_all_llama3_gptdriver.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t ours -d with_collision_all/far2near_error_2_error_middle_our_with_collision_all.pkl       -n far2near_error_2_error_middle_our_with_collision_all_llama3_our_method > log/far2near_error_2_error_middle_our_with_collision_all_llama3_our_method.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t gpt-driver -d with_collision_all/far2near_error_6_error_middle_our_with_collision_all.pkl -n far2near_error_6_error_middle_our_with_collision_all_llama3_gptdriver  > log/far2near_error_6_error_middle_our_with_collision_all_llama3_gptdriver.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t ours -d with_collision_all/far2near_error_6_error_middle_our_with_collision_all.pkl       -n far2near_error_6_error_middle_our_with_collision_all_llama3_our_method > log/far2near_error_6_error_middle_our_with_collision_all_llama3_our_method.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t gpt-driver -d with_collision_all/near2far_error_2_error_middle_our_with_collision_all.pkl -n near2far_error_2_error_middle_our_with_collision_all_llama3_gptdriver  > log/near2far_error_2_error_middle_our_with_collision_all_llama3_gptdriver.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t ours -d with_collision_all/near2far_error_2_error_middle_our_with_collision_all.pkl       -n near2far_error_2_error_middle_our_with_collision_all_llama3_our_method > log/near2far_error_2_error_middle_our_with_collision_all_llama3_our_method.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t gpt-driver -d with_collision_all/near2far_error_6_error_middle_our_with_collision_all.pkl -n near2far_error_6_error_middle_our_with_collision_all_llama3_gptdriver  > log/near2far_error_6_error_middle_our_with_collision_all_llama3_gptdriver.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t ours -d with_collision_all/near2far_error_6_error_middle_our_with_collision_all.pkl       -n near2far_error_6_error_middle_our_with_collision_all_llama3_our_method > log/near2far_error_6_error_middle_our_with_collision_all_llama3_our_method.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t gpt-driver -d with_collision_all/suddenly_appeal_error_middle_our_with_collision_all.pkl  -n suddenly_appeal_error_middle_our_with_collision_all_llama3_gptdriver  > log/suddenly_appeal_error_middle_our_with_collision_all_llama3_gptdriver.log 2>&1" \
#  "python -u incontext_generation.py -m llama3 -t ours -d with_collision_all/suddenly_appeal_error_middle_our_with_collision_all.pkl        -n suddenly_appeal_error_middle_our_with_collision_all_llama3_our_method > log/suddenly_appeal_error_middle_our_with_collision_all_llama3_our_method.log 2>&1" \
#echo "完成加异常前没有碰撞加异常侯碰撞+加前加后都没有碰撞+加前加后都碰撞的的数据集"

parallel -j 2 ::: \
  "python -u incontext_generation.py -p without_output_ana -m llama3 -t gpt-driver -d basic_dataset_random_point_modify_error_middle.pkl -n basic_dataset_random_point_modify_error_middle_llama3_gptdriver  > log/basic_dataset_random_point_modify_error_middle_llama3_gptdriver.log 2>&1" \
  "python -u incontext_generation.py -p without_output_ana -m llama3 -t ours       -d basic_dataset_random_point_modify_error_middle.pkl -n basic_dataset_random_point_modify_error_middle_llama3_our_method > log/basic_dataset_random_point_modify_error_middle_llama3_our_method.log 2>&1" \

echo "推理完成"
rm /data/code/xuzheling/llm_autodriver_error_handle/outputs/pkl/*text*
rm /data/code/xuzheling/llm_autodriver_error_handle/outputs/pkl/*jsonl

echo "删除无用输出"


echo "评估"
