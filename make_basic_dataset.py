import os
from tqdm import tqdm
from util import read_json, save_to_pkl, load_pkl_file
import random
from prompt_message import (
    system_message,
    generate_user_message,
    generate_assistant_message,
    generate_user_message_with_far2near_error_2error_point,
    generate_user_message_with_near2far_error_2error_point,
    generate_user_message_with_far2near_error_6error_point,
    generate_user_message_with_near2far_error_6error_point,
    generate_user_message_with_suddenly_appear_error_1error_point,
    example_message
)
random.seed(42)
f = read_json('data/split.json')
data = load_pkl_file('data/cached_nuscenes_info.pkl')
save_path = "data/our_dataset/"
evl_data = f.get('val')
train_data = f.get('train')
dataset_num = [
    "large",
    "middle",
    "small"
]
all_token = evl_data + train_data
data_preprocess = {
    "default": generate_user_message,
    "far2near_error_2_error": generate_user_message_with_far2near_error_2error_point,
    "far2near_error_6_error": generate_user_message_with_far2near_error_6error_point,
    "near2far_error_2_error": generate_user_message_with_near2far_error_2error_point,
    "near2far_error_6_error": generate_user_message_with_near2far_error_6error_point,
    "suddenly_appeal_error": generate_user_message_with_suddenly_appear_error_1error_point
}
for dataset_type in dataset_num:
    if dataset_type == 'small':
        all_token = evl_data[:int(len(evl_data) * 0.3)]
    elif dataset_type == 'middle':
        all_token = evl_data
    elif dataset_type == 'large':
        all_token = evl_data + train_data
    for dataset_name, data_process_func in data_preprocess.items():
        basic_dataset = dict()
        for index, token in tqdm(enumerate(all_token), total=len(all_token)):
            basic_dataset[token] = {
                "token": token,
                "input": data_process_func(data, token),
                "ground_truth": generate_assistant_message(data, token),
            }
        save_to_pkl(basic_dataset, os.path.join(save_path, f"basic_dataset_{dataset_name}_{dataset_type}.pkl"))


