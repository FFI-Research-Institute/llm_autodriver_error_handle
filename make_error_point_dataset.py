import os.path
import sys
from copy import deepcopy
import re
import copy
import math
import random

from util import load_pkl_file, save_to_pkl
import ast

random.seed(100)

data_size = "middle"

dataset = {
    "small": "basic_dataset_default_small.pkl",
    "middle": "basic_dataset_default_middle.pkl",
    "larger": "basic_dataset_default_large.pkl"
}

data_folder_path = "data/our_dataset/basic_dataset"


def parse_obstacle_line(line):
    # 更新正则表达式，允许行首有空白字符，并严格匹配整行
    pattern = r"^\s*-\s+([\w\.]+)\s+at\s+\(([-\d.]+),([-\d.]+)\)\.\s+Future trajectory:\s+(\[.*\])$"
    match = re.match(pattern, line)
    if match:
        obj_type = match.group(1)  # 障碍物名称
        x = float(match.group(2))  # 当前X坐标
        y = float(match.group(3))  # 当前Y坐标
        traj_str = match.group(4)  # 轨迹字符串

        try:
            # 将 'UN' 替换为 'None' 以便 ast.literal_eval 可以解析
            traj_str_clean = traj_str.replace('UN', 'None')

            # 使用 ast.literal_eval 解析轨迹字符串
            traj = ast.literal_eval(traj_str_clean)

            # 确保轨迹是列表且每个元素是元组
            if isinstance(traj, list) and all(isinstance(t, tuple) and len(t) == 2 for t in traj):
                # 将 'None' 保留为 None，数值保持为浮点数
                traj_processed = []
                for px, py in traj:
                    # px 和 py 可能是 float 或 None
                    px_val = float(px) if isinstance(px, (int, float)) else None
                    py_val = float(py) if isinstance(py, (int, float)) else None
                    traj_processed.append((px_val, py_val))
                traj = traj_processed
            else:
                raise ValueError("轨迹格式不正确")
        except Exception as e:
            print(f"解析轨迹时出错: {e} 在行: {line}")
            traj = []
            sys.exit(-1)

        return {
            "name": obj_type,
            "position": (x, y),
            "future_trajectory": traj
        }
    else:
        print(f"未匹配行: {line}")
        sys.exit(-1)
        # return None


def parse_text(text):
    lines = text.strip().splitlines()
    objects = []
    for line in lines:
        # 对每一行进行strip以去除行首和行尾的空白字符
        stripped_line = line.strip()
        if stripped_line:  # 跳过空行
            obj = parse_obstacle_line(stripped_line)
            if obj:
                objects.append(obj)
    return objects


def modify_obstacle_for_collision_with_curve(
        obstacle,
        vehicle_position=(0, 0),
        steps=6,
        move_fraction_range=(0.1, 0.3),
        angle_deviation_range=(-5, 5),  # 初始角度偏移范围（度）
        curve_angle_change_range=(-2, 2)  # 每步曲线角度变化范围（度）
):
    """
    修改障碍物的位置和未来轨迹，使其朝向车辆移动，并引入随机性和曲线轨迹以增加碰撞的可能性。

    :param obstacle: dict，包含 'name', 'position', 'future_trajectory'
    :param vehicle_position: tuple，本车的位置，默认 (0, 0)
    :param steps: int，未来轨迹的步数，默认 6
    :param move_fraction_range: tuple，每步移动距离占当前位置与车辆位置距离的比例范围，默认 (0.1, 0.3)
    :param angle_deviation_range: tuple，初始移动方向的随机偏移范围（度），默认 (-5, 5)
    :param curve_angle_change_range: tuple，每步移动方向的随机偏移变化范围（度），默认 (-2, 2)
    :return: dict，修改后的障碍物字典
    """
    modified_obstacle = copy.deepcopy(obstacle)

    # 当前障碍物位置
    obs_x, obs_y = modified_obstacle['position']
    veh_x, veh_y = vehicle_position

    # 计算从障碍物到车辆的向量
    vector_x = veh_x - obs_x
    vector_y = veh_y - obs_y
    distance = math.hypot(vector_x, vector_y)

    if distance == 0:
        print("障碍物已在车辆位置，无法进一步移动。")
        return modified_obstacle

    # 计算基础角度（弧度）
    base_angle = math.atan2(vector_y, vector_x)

    # 随机选择初始角度偏移
    initial_angle_deviation_deg = random.uniform(*angle_deviation_range)
    initial_angle_deviation_rad = math.radians(initial_angle_deviation_deg)

    # 计算新的初始角度
    current_angle = base_angle + initial_angle_deviation_rad

    # 计算单位向量
    unit_vector_x = math.cos(current_angle)
    unit_vector_y = math.sin(current_angle)

    # 随机选择初始移动比例
    move_fraction = random.uniform(*move_fraction_range)

    # 修改当前障碍物位置，向车辆移动一定比例的距离
    move_distance = move_fraction * distance
    new_x = obs_x + unit_vector_x * move_distance
    new_y = obs_y + unit_vector_y * move_distance
    modified_obstacle['position'] = (round(new_x, 2), round(new_y, 2))

    # 生成未来轨迹，逐步靠近车辆，并引入曲线
    future_trajectory = []
    current_x, current_y = new_x, new_y

    # 计算初始实际距离
    current_distance = math.hypot(veh_x - current_x, veh_y - current_y)

    for step in range(steps):
        if current_distance <= 0:
            break  # 已经到达或超过车辆位置

        # 随机选择每步移动的比例
        step_move_fraction = random.uniform(*move_fraction_range)

        # 计算本步移动距离
        step_move_distance = step_move_fraction * current_distance if current_distance > 0 else 0

        # 随机选择曲线角度变化
        angle_change_deg = random.uniform(*curve_angle_change_range)
        angle_change_rad = math.radians(angle_change_deg)

        # 更新当前移动角度
        current_angle += angle_change_rad

        # 计算新的单位向量
        unit_vector_x = math.cos(current_angle)
        unit_vector_y = math.sin(current_angle)

        # 更新位置
        current_x += unit_vector_x * step_move_distance
        current_y += unit_vector_y * step_move_distance

        # 重新计算实际距离
        current_distance = math.hypot(veh_x - current_x, veh_y - current_y)

        # 四舍五入到两位小数
        future_trajectory.append((round(current_x, 2), round(current_y, 2)))

    modified_obstacle['future_trajectory'] = future_trajectory

    return modified_obstacle


def modify_obstacle_for_error_point(obstacle):
    """
    随机修改障碍物未来轨迹中的两个点，给其增加异常。

    参数:
        obstacle (dict): 包含 'name', 'position', 'future_trajectory' 的字典。

    返回:
        dict: 修改后的障碍物字典。
    """
    # 复制未来轨迹以避免修改原始数据
    future_trajectory = obstacle.get('future_trajectory', []).copy()

    # 检查未来轨迹点的数量
    if len(future_trajectory) < 2:
        raise ValueError("future_trajectory 中的点数不足两个，无法进行修改。")

    # 随机选择两个不同的索引
    indices = random.sample(range(len(future_trajectory)), 2)

    for idx in indices:
        x, y = future_trajectory[idx]

        # 修改 X 坐标
        if x is not None:
            change_x = random.randint(50, 100)
            # 随机决定增加还是减少
            if random.choice([True, False]):
                new_x = x + change_x
            else:
                new_x = x - change_x
        else:
            new_x = None

        # 修改 Y 坐标
        if y is not None:
            change_y = random.randint(50, 100)
            # 随机决定增加还是减少
            if random.choice([True, False]):
                new_y = y + change_y
            else:
                new_y = y - change_y
        else:
            new_y = None

        # 更新轨迹点
        future_trajectory[idx] = (new_x, new_y)
        print(f"修改了索引 {idx} 的点: 原点 ({x}, {y}) -> 新点 ({new_x}, {new_y})")

    # 更新障碍物的未来轨迹
    obstacle['future_trajectory'] = future_trajectory

    return obstacle


def replace_none_with_previous(data):
    # 获取初始位置
    prev_x, prev_y = data['position']

    # 遍历未来轨迹
    for i, (x, y) in enumerate(data['future_trajectory']):
        # 检查并替换 x 坐标
        if x is None:
            x = prev_x
            print(f"替换 future_trajectory[{i}][0] 为前一个 x: {x}")

        # 检查并替换 y 坐标
        if y is None:
            y = prev_y
            print(f"替换 future_trajectory[{i}][1] 为前一个 y: {y}")

        # 更新轨迹中的坐标
        data['future_trajectory'][i] = (x, y)

        # 更新前一个有效的坐标点
        prev_x, prev_y = x, y

    return data


def add_error_toward_our_car(ob: str) -> str:
    objects = parse_text(ob)

    if len(objects) == 0:
        return ob
    try:
        choose_index = random.randint(0, len(objects) - 1)
    except Exception as e:
        choose_index = 0

    filter_ob = replace_none_with_previous(objects[choose_index])
    error_ob = modify_obstacle_for_collision_with_curve(filter_ob)
    objects[choose_index] = deepcopy(error_ob)

    rt_str = ""
    for x in objects:
        position = str(x['position']).replace(' ', '')
        future_trajectory = str(x['future_trajectory']).replace(' ', '')
        this_str = f"- {x['name']} at {position}. Future trajectory: {future_trajectory}\n"
        rt_str += this_str
    return f"\n{rt_str}"


def add_random_error_points(ob: str) -> str:
    objects = parse_text(ob)

    if len(objects) == 0:
        return ob
    try:
        choose_index = random.randint(0, len(objects) - 1)
    except Exception as e:
        choose_index = 0
    filter_ob = replace_none_with_previous(objects[choose_index])
    error_ob = modify_obstacle_for_error_point(filter_ob)
    objects[choose_index] = deepcopy(error_ob)

    rt_str = ""
    for x in objects:
        position = str(x['position']).replace(' ', '')
        future_trajectory = str(x['future_trajectory']).replace(' ', '')
        this_str = f"- {x['name']} at {position}. Future trajectory: {future_trajectory}\n"
        rt_str += this_str
    return f"\n{rt_str}"


if __name__ == "__main__":
    datas = load_pkl_file(os.path.join(data_folder_path, dataset[data_size]))
    new_dataset_error_toward_our_car = deepcopy(datas)
    new_dataset_random_error_points = deepcopy(datas)
    for hash_key, data in datas.items():
        lines = data['input'].splitlines()
        obstacles = ""
        for line in lines:
            if "Perception and Prediction" in line:
                continue
            if "Ego-States" in line:
                break
            obstacles += f"{line}\n"
        print(f"no error:\n{data['input']}")
        new_input_random_error_points = data['input'].replace(obstacles, add_random_error_points(obstacles))
        new_dataset_random_error_points[hash_key]['input'] = new_input_random_error_points
        print(f"add error:\n{new_dataset_random_error_points[hash_key]['input']}")
        break
        # new_input_error_toward_our_car = data['input'].replace(obstacles, add_error_toward_our_car(obstacles))
        # new_dataset_error_toward_our_car[hash_key]['input'] = new_input_error_toward_our_car
        # print(f"add error:\n{new_dataset_error_toward_our_car[hash_key]['input']}")
    # save_to_pkl(new_dataset_error_toward_our_car, os.path.join(data_folder_path, f"basic_dataset_modify_toward_our_car_error_{data_size}.pkl"))
    save_to_pkl(new_dataset_random_error_points,
                os.path.join(data_folder_path, f"basic_dataset_random_point_modify_error_{data_size}.pkl"))
