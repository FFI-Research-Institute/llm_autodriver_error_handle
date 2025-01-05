import pickle
import torch
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple
from util import *


def check_data_format(data):
    for item in data:
        # 检查每个元素是否是一个长度为2的列表或数组
        if not isinstance(item, (list, np.ndarray)) or len(item) != 2:
            return False
        # 检查每个元素的值是否为浮点数
        if not all(isinstance(x, (float, np.float32, np.float64)) for x in item):
            return False
    return True


def load_pkl(file_path):
    """
    加载并打印指定路径的pkl文件内容。

    参数:
    file_path (str): pkl文件的路径
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。请检查文件路径是否正确。")
    except pickle.UnpicklingError:
        print("错误: 无法解序列化文件。请确保文件是有效的pkl格式。")
    except Exception as e:
        print(f"发生错误: {e}")


def meters_to_pixels(x: float, y: float, x_min: float, y_min: float, resolution: float) -> Tuple[float, float]:
    """
    将米转换为像素坐标。

    参数：
        x (float): 米单位的x坐标。
        y (float): 米单位的y坐标。
        x_min (float): 地图x轴的最小值。
        y_min (float): 地图y轴的最小值。
        resolution (float): 每个像素代表的米数。

    返回：
        (float, float): 对应的像素坐标 (col, row)。
    """
    col = (x - x_min) / resolution
    row = (y - y_min) / resolution
    return col, row


def draw_vehicle(ax, center_x: float, center_y: float, W_pixels: float, H_pixels: float, color: str, label: str = None):
    """
    在地图上绘制车辆的矩形表示。

    参数：
        ax (matplotlib.axes.Axes): 绘图的坐标轴。
        center_x (float): 车辆中心的像素x坐标。
        center_y (float): 车辆中心的像素y坐标。
        W_pixels (float): 车辆宽度的像素数。
        H_pixels (float): 车辆高度的像素数。
        color (str): 矩形颜色。
        label (str, optional): 标签名称，仅在第一次调用时添加标签。
    """
    lower_left_x = center_x - W_pixels / 2
    lower_left_y = center_y - H_pixels / 2
    rect = patches.Rectangle(
        (lower_left_x, lower_left_y), W_pixels, H_pixels,
        linewidth=2, edgecolor=color, facecolor='none',
        label=label if label else None
    )
    ax.add_patch(rect)


def create_animation(
        map_tensor: torch.Tensor,
        gt_traj: torch.Tensor,
        pred_traj: torch.Tensor,
        video_filename: str = 'output_video.mp4',
        fps: int = 1
):
    """
    创建一个动画展示地图变动，并绘制本车的真实轨迹和预测轨迹。

    参数:
        map_tensor (torch.Tensor): 形状为 [1, 7, 200, 200] 的布尔型张量，表示7帧200x200像素的地图。
        gt_traj (torch.Tensor): 本车的真实轨迹，形状为 [1, 7, 2]，单位为米。
        pred_traj (torch.Tensor): 本车的预测轨迹，形状为 [1, 7, 2]，单位为米。
        video_filename (str): 输出视频文件名，如 "output_video.mp4"。
        fps (int): 视频帧率，默认为1帧每秒。
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import imageio
    from matplotlib import animation

    # 检查输入形状
    assert map_tensor.shape == (1, 7, 200, 200), f"map_tensor的形状必须为 [1, 7, 200, 200], 但得到 {map_tensor.shape}"
    assert gt_traj.shape == (1, 7, 2), f"gt_traj的形状必须为 [1, 7, 2], 但得到 {gt_traj.shape}"
    assert pred_traj.shape == (1, 7, 2), f"pred_traj的形状必须为 [1, 7, 2], 但得到 {pred_traj.shape}"

    # 地图参数
    x_min, x_max, resolution = -50.0, 50.0, 0.5  # 与 PlanningMetric 类一致
    y_min, y_max = -50.0, 50.0
    W_m, H_m = 1.85, 4.084  # 车辆尺寸（米）
    W_pixels, H_pixels = W_m / resolution, H_m / resolution  # 车辆尺寸（像素）

    # 将地图转换为NumPy数组
    map_np = map_tensor.squeeze(0).cpu().numpy()  # 形状为 [7, 200, 200]

    # 将轨迹转换为NumPy数组
    gt_traj_np = gt_traj.squeeze(0).cpu().numpy()  # 形状为 [7, 2]
    pred_traj_np = pred_traj.squeeze(0).cpu().numpy()  # 形状为 [7, 2]

    # 将米转换为像素坐标
    gt_pixels = [meters_to_pixels(x, y, x_min, y_min, resolution) for x, y in gt_traj_np]
    pred_pixels = [meters_to_pixels(x, y, x_min, y_min, resolution) for x, y in pred_traj_np]

    # 反转y轴以适应图像显示（图像y轴从上到下递增）
    gt_pixels = [(col, 200 - row) for col, row in gt_pixels]
    pred_pixels = [(col, 200 - row) for col, row in pred_pixels]

    # 创建视频写入器
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 6))

    # 设置坐标轴
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.axis('off')  # 关闭坐标轴

    # 初始化地图显示
    im = ax.imshow(map_np[0], cmap='gray', origin='upper', extent=[0, 200, 0, 200])

    # 初始化散点和车辆矩形
    gt_scatter = ax.scatter([], [], c='green', s=100, label='Ground Truth')
    pred_scatter = ax.scatter([], [], c='blue', s=100, label='Prediction')
    gt_vehicle = patches.Rectangle((0, 0), W_pixels, H_pixels, linewidth=2, edgecolor='green', facecolor='none',
                                   label='GT Vehicle')
    pred_vehicle = patches.Rectangle((0, 0), W_pixels, H_pixels, linewidth=2, edgecolor='blue', facecolor='none',
                                     label='Predicted Vehicle')
    ax.add_patch(gt_vehicle)
    ax.add_patch(pred_vehicle)

    # 添加图例（只添加一次）
    ax.legend(loc='upper right')

    # 更新函数
    def update(frame):
        # 更新地图
        im.set_data(map_np[frame])

        # 更新真实轨迹点
        gt_col, gt_row = gt_pixels[frame]
        gt_scatter.set_offsets([[gt_col, gt_row]])

        # 更新预测轨迹点
        pred_col, pred_row = pred_pixels[frame]
        pred_scatter.set_offsets([[pred_col, pred_row]])

        # 更新车辆矩形位置
        gt_vehicle.set_xy((gt_col - W_pixels / 2, gt_row - H_pixels / 2))
        pred_vehicle.set_xy((pred_col - W_pixels / 2, pred_row - H_pixels / 2))

        # 更新标题
        ax.set_title(f"Frame {frame + 1}")

        return [im, gt_scatter, pred_scatter, gt_vehicle, pred_vehicle]

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=7, blit=True)

    # 保存动画为视频
    ani.save(video_filename, writer=writer)

    plt.close(fig)
    print(f"视频已保存为 {video_filename}")


def add_initial_zero(traj_array):
    """
    在轨迹的第一个时间步添加一个 [0., 0.]。

    参数：
        traj_array (np.ndarray): 输入轨迹数组，形状为 (batch_size, n_future, 2)

    返回：
        np.ndarray: 在时间维度前添加 [0., 0.] 后的轨迹数组，形状为 (batch_size, n_future + 1, 2)
    """
    # 获取批次大小和坐标维度
    batch_size, n_future, coord_dim = traj_array.shape

    # 创建一个全零的数组，形状为 (batch_size, 1, 2)
    zero_frame = np.zeros((batch_size, 1, coord_dim), dtype=traj_array.dtype)

    # 在时间维度（轴=1）上拼接
    new_traj = np.concatenate((zero_frame, traj_array), axis=1)

    return new_traj


def add_batch_dimension(traj_array):
    """
    将形状为 (6, 2) 的数组转换为 (1, 6, 2)，添加一个批次维度。

    参数：
        traj_array (np.ndarray): 输入轨迹数组，形状为 (6, 2)

    返回：
        np.ndarray: 添加批次维度后的数组，形状为 (1, 6, 2)
    """
    if traj_array.ndim == 2 and traj_array.shape == (6, 2):
        return np.expand_dims(traj_array, axis=0)
    else:
        raise ValueError("输入数组的形状必须为 (6, 2)")


if __name__ == "__main__":
    map_pkl_path = 'evl/gt/planing_gt_segmentation_val'
    gt_pkl_path = 'evl/gt/gt_traj.pkl'
    all_pkl_files = get_pkl_files("outputs/pkl/")
    for predict_pkl_path in all_pkl_files:
        # predict_pkl_path = 'outputs/pkl/far2near_error_llama3.2_gptdriver_100.pkl'
        folder_path = f"outputs/videos/{(predict_pkl_path.split('/')[-1])[:-4]}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        keys = load_pkl(predict_pkl_path).keys()
        for token in keys:
            gt_occ_map = load_pkl(map_pkl_path)[token]
            gt_trajs = torch.tensor(add_initial_zero(load_pkl(gt_pkl_path)[token]))
            predict_trajs = torch.tensor(add_initial_zero(add_batch_dimension(load_pkl(predict_pkl_path)[token])))
            print(gt_occ_map.shape)
            print(50 * '*')
            print(gt_trajs.shape)
            print(f"now the truth traj:")
            print(gt_trajs)
            print(50 * '*')
            print(predict_trajs.shape)
            print(f"now the predict traj:")
            print(predict_trajs)
            print(50 * '*')

            create_animation(
                map_tensor=gt_occ_map,
                gt_traj=gt_trajs,
                pred_traj=predict_trajs,
                video_filename=f"outputs/videos/{(predict_pkl_path.split('/')[-1])[:-4]}/{token}.mp4",
                fps=1  # 每秒1帧，可以根据需要调整
            )
