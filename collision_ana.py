import csv
from util import load_pkl_file, get_pkl_files
from copy import deepcopy

file_type = "llama"

files_path = "outputs/collision_set"
pkl_files = get_pkl_files(files_path)
pkl_files = [f for f in pkl_files if file_type in f]
pkl_files.sort()

default_files = [x for x in pkl_files if "default" in x]
far2near_error_file = [x for x in pkl_files if "far2near" in x]
near2far_error_file = [x for x in pkl_files if "near2far" in x]
suddenly_appeal_file = [x for x in pkl_files if "suddenly" in x]

result = dict()

for file_list in [default_files, far2near_error_file, near2far_error_file, suddenly_appeal_file]:
    for file_name in file_list:
        hash_set = load_pkl_file(file_name)
        for hash_key in hash_set:
            if hash_key not in result:
                result[hash_key] = {x.split('/')[-1]: 0 for x in pkl_files}
                # result[hash_key] = dict()
            result[hash_key][file_name.split('/')[-1]] = 1
print(result)

fields = list(next(iter(result.values())).keys())

# 打开CSV文件进行写入
with open(f'outputs/{file_type}_collision_ana.csv', mode='w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)

    # 写入表头（第一行）
    header = ['ID'] + fields
    csv_writer.writerow(header)

    # 写入每一行数据
    for key, value in result.items():
        row = [key] + [value.get(field, '') for field in fields]
        csv_writer.writerow(row)

print("CSV文件已保存。")
# # 打印交集（两个集合都有碰撞的数据）
# print(f"{10*'*'}两个都碰撞了{10*'*'}")
# intersection = no_error & with_error
# for i in intersection:
#     print(i)
#
# # 打印只在有异常数据的集合中出现的数据（有异常数据的碰撞，但是没有异常数据的没有碰撞）
# print(f"{10*'*'}有异常数据的碰撞但是没有异常数据的没有碰撞{10*'*'}")
# only_with_error = with_error - no_error
# for i in only_with_error:
#     print(i)
#
# # 打印只在没有异常数据的集合中出现的数据（没有异常数据的碰撞但是有异常数据的没有碰撞）
# print(f"{10*'*'}没有异常数据的碰撞但是有异常数据的没有碰撞{10*'*'}")
# only_no_error = no_error - with_error
# for i in only_no_error:
#     print(i)
