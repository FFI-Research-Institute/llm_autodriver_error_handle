import csv
import os.path

from util import save_to_pkl, load_pkl_file

# 打开文件并读取所有数据
with open('outputs/llama_collision_ana.csv', mode='r', encoding='utf-8') as file:
    csv_dict_reader = csv.DictReader(file)

    # 读取所有行数据
    rows = list(csv_dict_reader)

# 获取header，从第三列开始
header = csv_dict_reader.fieldnames[2:]
# print(csv_dict_reader)

result_dict = {
    "without_collision_all": dict(),
    "with_collision_all": dict(),
    "only_collision_after_error": dict()
}
for data_type in header:
    # print(data_type)
    data_info = [0, 0, 0]
    hash_key_without_collision_all = list()
    hash_key_with_collision_all = list()
    hash_key_with_only_collision_after_error = list()
    for row in rows:
        if ((row["basic_dataset_default_middle_llama3_gptdriver_baseline.pkl"] == '0' and row[data_type] == '1') or
            (row["basic_dataset_default_middle_llama3_gptdriver_baseline.pkl"] == '0' and row[data_type] == '0') or
            (row["basic_dataset_default_middle_llama3_gptdriver_baseline.pkl"] == '1' and row[data_type] == '1')):
            if row["basic_dataset_default_middle_llama3_gptdriver_baseline.pkl"] == '0' and row[data_type] == '1':
                hash_key_with_only_collision_after_error.append(row["ID"])
                data_info[0] += 1
            if row["basic_dataset_default_middle_llama3_gptdriver_baseline.pkl"] == '0' and row[data_type] == '0':
                data_info[1] += 1
            if row["basic_dataset_default_middle_llama3_gptdriver_baseline.pkl"] == '1' and row[data_type] == '1':
                data_info[2] += 1
            hash_key_with_collision_all.append(row["ID"])
            if ((row["basic_dataset_default_middle_llama3_gptdriver_baseline.pkl"] == '0' and row[data_type] == '1') or
                (row["basic_dataset_default_middle_llama3_gptdriver_baseline.pkl"] == '0' and row[data_type] == '0')):
                hash_key_without_collision_all.append(row["ID"])
    result_dict["without_collision_all"][data_type] = hash_key_without_collision_all.copy()
    result_dict["with_collision_all"][data_type] = hash_key_with_collision_all.copy()
    result_dict["only_collision_after_error"][data_type] = hash_key_with_only_collision_after_error.copy()
    print(f"{data_type}中，原来没有撞加了异常装了：{data_info[0]}，都没有撞：{data_info[1]}，都撞了：{data_info[2]}")

print("没有包含都碰撞")
for key, value in result_dict["without_collision_all"].items():
    print(f"{key[:-30]}: num: {len(value)}")
print("包含都碰撞")
for key, value in result_dict["with_collision_all"].items():
    print(f"{key[:-30]}: num: {len(value)}")

for data_type in ["without_collision_all", "with_collision_all", "only_collision_after_error"]:
    for file_name, values in result_dict[data_type].items():
        our_dataset = dict()
        # print(file_name[:-30])
        all_data = load_pkl_file(os.path.join("data/our_dataset/basic_dataset", file_name[:-30] + '.pkl'))
        for hash_key in values:
            our_dataset[hash_key] = all_data[hash_key]
            # print(our_dataset)
        save_to_pkl(our_dataset, os.path.join(f"data/our_dataset/our_select/{data_type}",
                                              f"{file_name[14:-30]}_our_{data_type}.pkl"))
