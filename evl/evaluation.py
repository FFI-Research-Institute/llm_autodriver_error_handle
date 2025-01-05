import torch
from torch import Tensor
from tqdm import tqdm
import pickle
import json
from pathlib import Path
import os
from metric import PlanningMetric


def planning_evaluation(pred_trajs_dict, subset=None, only_vehicle=True):
    future_second = 3
    ts = future_second * 2
    device = torch.device('cpu')

    metric_planning_val = PlanningMetric(ts).to(device)

    if only_vehicle:
        with open('gt/planing_gt_segmentation_val', 'rb') as f:
            gt_occ_map = pickle.load(f)
        for token in gt_occ_map.keys():
            if not isinstance(gt_occ_map[token], torch.Tensor):
                gt_occ_map[token] = torch.tensor(gt_occ_map[token])
            gt_occ_map[token] = torch.flip(gt_occ_map[token], [-1])
    else:
        with open('eval_share/gt/vad_gt_seg.pkl', 'rb') as f:
            gt_occ_map_woP = pickle.load(f)
        for token in gt_occ_map_woP.keys():
            if not isinstance(gt_occ_map_woP[token], torch.Tensor):
                gt_occ_map_woP[token] = torch.tensor(gt_occ_map_woP[token])
            gt_occ_map_woP[token] = torch.flip(gt_occ_map_woP[token], [-1])
            gt_occ_map_woP[token] = torch.flip(gt_occ_map_woP[token], [-2])
        gt_occ_map = gt_occ_map_woP

    with open('gt/gt_traj.pkl', 'rb') as f:
        gt_trajs_dict = pickle.load(f)

    with open('gt/gt_traj_mask.pkl', 'rb') as f:
        gt_trajs_mask_dict = pickle.load(f)

    if subset:
        test_tokens = subset

    for index, token in enumerate(tqdm(gt_trajs_dict.keys())):
        gt_trajectory = torch.tensor(gt_trajs_dict[token])
        gt_trajectory = gt_trajectory.to(device)

        gt_traj_mask = torch.tensor(gt_trajs_mask_dict[token])
        gt_traj_mask = gt_traj_mask.to(device)

        try:
            output_trajs = torch.tensor(pred_trajs_dict[token])
            output_trajs = output_trajs.reshape(gt_traj_mask.shape)
        except Exception as e:
            continue
        output_trajs = output_trajs.to(device)

        occupancy: Tensor = gt_occ_map[token]
        occupancy = occupancy.to(device)

        if output_trajs.shape[1] % 2:  # in case the current time is inculded
            output_trajs = output_trajs[:, 1:]

        if occupancy.shape[1] % 2:  # in case the current time is inculded
            occupancy = occupancy[:, 1:]

        if gt_trajectory.shape[1] % 2:  # in case the current time is inculded
            gt_trajectory = gt_trajectory[:, 1:]

        if gt_traj_mask.shape[1] % 2:  # in case the current time is inculded
            gt_traj_mask = gt_traj_mask[:, 1:]

        metric_planning_val(output_trajs[:, :ts], gt_trajectory[:, :ts], occupancy[:, :ts], token, gt_traj_mask)

    results = {}
    error_token, scores = metric_planning_val.compute()
    print(scores)
    for i in range(future_second):
        for key, value in scores.items():
            results['plan_' + key + '_{}s'.format(i + 1)] = value[:(i + 1) * 2].mean()

    # Print results in table
    print(f"gt collision: {metric_planning_val.gt_collision}")
    headers = ["Method", "L2 (m)", "Collision (%)"]
    sub_headers = ["1s", "2s", "3s", "Avg."]
    method = (
        "DriveAgent", "{:.2f}".format(scores["L2"][1]), "{:.2f}".format(scores["L2"][3]),
        "{:.2f}".format(scores["L2"][5]), \
        "{:.2f}".format((scores["L2"][1] + scores["L2"][3] + scores["L2"][5]) / 3.), \
        "{:.2f}".format(scores["obj_box_col"][1] * 100), \
        "{:.2f}".format(scores["obj_box_col"][3] * 100), \
        "{:.2f}".format(scores["obj_box_col"][5] * 100), \
        "{:.2f}".format(100 * (scores["obj_box_col"][1] + scores["obj_box_col"][3] + scores["obj_box_col"][5]) / 3.))
    print("\n")
    print("UniAD evaluation:")
    print("{:<15} {:<20} {:<20}".format(*headers))
    print("{:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format("", *sub_headers, *sub_headers))
    print("{:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format(*method))

    method = ("DriveAgent", "{:.2f}".format(results["plan_L2_1s"]), "{:.2f}".format(results["plan_L2_2s"]),
              "{:.2f}".format(results["plan_L2_3s"]), \
              "{:.2f}".format((results["plan_L2_1s"] + results["plan_L2_2s"] + results["plan_L2_3s"]) / 3.), \
              "{:.2f}".format(results["plan_obj_box_col_1s"] * 100),
              "{:.2f}".format(results["plan_obj_box_col_2s"] * 100),
              "{:.2f}".format(results["plan_obj_box_col_3s"] * 100), \
              "{:.2f}".format(((results["plan_obj_box_col_1s"] + results["plan_obj_box_col_2s"] + results[
                  "plan_obj_box_col_3s"]) / 3) * 100))
    print("\n")
    print("STP-3 evaluation:")
    print("{:<15} {:<20} {:<20}".format(*headers))
    print("{:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format("", *sub_headers, *sub_headers))
    print("{:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format(*method))

    # for col in error_token:
    #     print(col)

    return (results["plan_L2_1s"] + results["plan_L2_2s"] + results["plan_L2_3s"]) / 3., ((results[
                                                                                               "plan_obj_box_col_1s"] +
                                                                                           results[
                                                                                               "plan_obj_box_col_2s"] +
                                                                                           results[
                                                                                               "plan_obj_box_col_3s"]) / 3) * 100, error_token


def load_pred_trajs_from_file(path):
    with open(path, "rb") as f:
        pred_trajs_dict = pickle.load(f)
    return pred_trajs_dict


def save_set_to_pkl(my_set, filename):
    with open(filename, 'wb') as f:
        pickle.dump(my_set, f)


def get_pkl_files(folder_path):
    # 获取文件夹下所有的文件
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    pkl_files = [f for f in pkl_files if 'text' not in f]
    pkl_files.sort()
    # 返回完整的文件路径
    pkl_file_paths = [os.path.join(folder_path, f) for f in pkl_files]
    return pkl_file_paths


if __name__ == "__main__":
    """
    This evaluation function will report both the STP-3 metric (avg over avg) and the UniAD metric. 
    Since UniAD only considers the vehicle category when generating ground truth occupancy, while ST-P3 considers both the vehicle and pedestrian categories.
    If you want to report the STP-3 metric, please set only_vehicle=False.
    if you want to report the UniAD metric, please set only_vehicle=True.
    """
    from util import load_pkl_file, save_to_pkl

    file_path = "../outputs/pkl/"
    all_files = get_pkl_files(file_path)
    all_files = [x for x in all_files if "llama3" in x]
    all_files.sort()
    have_evl = load_pkl_file("have_evl.pkl")
    for file in all_files:
        if file in have_evl:
            pass
            # continue
        else:
            have_evl.append(file)
        print(f"{50 * '*'}")
        print(file)
        pred_trajs_dict = load_pred_trajs_from_file(file)
        _, _, error_tokens = planning_evaluation(pred_trajs_dict, subset=None, only_vehicle=True)
        save_set_to_pkl(error_tokens, f"../outputs/collision_set/{(file.split('/')[-1])[:-4]}.pkl")
    save_to_pkl(have_evl, "have_evl.pkl")