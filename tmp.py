import pickle

# 文件路径
file_path = '/data/code/xuzheling/llm_autodriver_error_handle/data/our_dataset/our_select/only_collision_after_error/far2near_error_2_error_middle_our_only_collision_after_error.pkl'

try:
    # 以二进制读取模式打开文件
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # 打印内容
    print(data["3f2adb9db8ab428abd54009707d46992"]["ground_truth"])

except FileNotFoundError:
    print(f"文件未找到：{file_path}")
except pickle.UnpicklingError:
    print("无法解序列化文件内容。请确保文件是有效的 pickle 文件。")
except Exception as e:
    print(f"发生错误：{e}")
