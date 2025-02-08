from util import load_pkl_file

a = load_pkl_file("data/our_dataset/basic_dataset/basic_dataset_random_point_modify_error_middle.pkl")
print(a.keys())
print(a['2e137fee62e241aca2966876af1e201f']['input'])
# print(len(a))
# b = load_pkl_file("outputs/pkl/basic_dataset_default_middle_llama3_gptdriver.pkl")
# print(len(b))
# c = load_pkl_file("data/our_dataset/basic_dataset/basic_dataset_default_middle.pkl")
# print(len(c))