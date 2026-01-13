# Copyright (c) OpenMMLab. All rights reserved.
import os
from os import path as osp
import numpy as np


import pickle

from pytorchse3.se3 import se3_log_map, se3_exp_map

from llava.model import *

import torch
from mmengine import dump, load

from PIL import Image, ImageDraw

import argparse
from tqdm import tqdm

import cv2
from sklearn.cluster import KMeans

import trimesh.transformations as tra
import open3d as o3d   # 追加
from llava.train.visiual import merge_grasp
from llava.train.llm import read_grasp_part

from scipy.spatial.transform import Rotation as R




# x: right, y: up, z: forward

# front：fov_degrees = 0, f = 1, R = [[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]], t = [0.0, -0.5, 1.0]
# bottom right: fov_degrees = 90, f = 1, R = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]], t = [-0.5, 0.0, 0.5]
# BEV: fov_degrees = 180, f = 1.5, R = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]], t = [0, 0.0, 0.5]
# top：fov_degrees = 270, f = 2.0, R = [[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]], t = [0.0, 0.5, 1.0]

max_num = 100

MAX_WIDTH = 0.202   # maximum width of gripper 2F-140
width, height = 336, 336
seperate_num = 100000

def project_points(pc_xyz, K, ext):
    pc_homogeneous = torch.cat((pc_xyz, torch.ones(pc_xyz.shape[0], 1, dtype=torch.float32)), dim=1)
    proj_matrix = K @ ext
    proj_points = proj_matrix @ pc_homogeneous.t()
    proj_points[:2, :] = proj_points[:2, :] / (proj_points[2, :]+1e-4) 
    return proj_points.t()

def get_gripper_control_points():
    return np.array([
        [-0.10, 0, 0, 1],
        [-0.03, 0, 0, 1],
        [-0.03, 0.07, 0, 1],
        [0.03, 0.07, 0, 1],
        [-0.03, -0.07, 0, 1],
        [0.03, -0.07, 0, 1]])

def points2deprgb(points, pc_rgb, height, width):
    points[:, 0] = points[:, 0] * width
    points[:, 1] = points[:, 1] * height
    depth_map = torch.zeros((height, width), dtype=torch.float32)
    img_rgb = torch.ones((height, width, 3), dtype=torch.float32) #背景全为白色
    # img_rgb = torch.zeros((height, width, 3), dtype=torch.float32) #背景全为黑色
    coor = torch.round(points[:, :2])
    depth = points[:, 2]
    kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
        coor[:, 1] >= 0) & (coor[:, 1] < height)
    coor, depth, rgb = coor[kept1], depth[kept1], pc_rgb[kept1]
    ranks = coor[:, 0] + coor[:, 1] * width
    sort = (ranks + depth / 100.).argsort()
    coor, depth, rgb, ranks = coor[sort], depth[sort], rgb[sort], ranks[sort]

    kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
    kept2[1:] = (ranks[1:] != ranks[:-1])
    coor, depth, rgb = coor[kept2], depth[kept2], rgb[kept2]
    coor = coor.to(torch.long)
    depth_map[coor[:, 1], coor[:, 0]] = depth
    img_rgb[coor[:, 1], coor[:, 0]] = rgb

    depth_map = (depth_map * 255).to(torch.uint8)
    img_rgb = (img_rgb * 255).to(torch.uint8)
    return depth_map, img_rgb


def pruning_grasps(grasps, rot_weight=1.0, trans_weight=3.0, w_weight=0.1, max_num=100):

    rots = grasps[:, :3]
    trans = grasps[:, 3:6]
    # widths = grasps[:, 6]

    rot_distance = torch.norm(rots[:, None, :] - rots[None, :, :], dim=2)
    trans_distance = torch.norm(trans[:, None, :] - trans[None, :, :], dim=2)
    # width_distance = torch.abs(widths[:, None] - widths[None, ])

    # similarity_matrix = rot_weight * rot_distance + trans_weight * trans_distance + w_weight * width_distance
    similarity_matrix = rot_weight * rot_distance + trans_weight * trans_distance


    kmeans = KMeans(n_clusters=max_num, random_state=0, n_init='auto').fit(grasps.detach().numpy())
    cluster_labels = kmeans.labels_

    selected_grasps, selected_ids = [], []
    for cluster_id in range(max_num):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 1:
            selected_grasps.append(grasps[cluster_indices[0]])
            selected_ids.append(cluster_indices[0])
            continue
       
        grasps_incluster = grasps[cluster_indices]
        similarity_sum = torch.sum(similarity_matrix[cluster_indices, :][:, cluster_indices], dim=1)
        min_similarity_index = torch.argmin(similarity_sum, dim=0)
        selected_grasps.append(grasps_incluster[min_similarity_index])
        selected_ids.append(int(cluster_indices[min_similarity_index]))
    selected_grasps_tensor = torch.stack(selected_grasps)
    selected_ids = torch.Tensor(selected_ids).to(torch.int64)
    return selected_grasps_tensor

def rotate_Twc_around_z(pivot, T_wc_base, angle_rad):
    Rz_delta = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0,                  0,                 1],
    ], dtype=np.float64)

    R_old = T_wc_base[:3, :3]
    C_old = T_wc_base[:3, 3]

    R_new = Rz_delta @ R_old
    C_new = pivot + Rz_delta @ (C_old - pivot)

    T_new = np.eye(4, dtype=np.float64)
    T_new[:3, :3] = R_new
    T_new[:3, 3]  = C_new
    return T_new

def rotate_extrinsic_world2cam_around_z(pivot, E_w2c, angle_rad):
    # E_w2c: World->Cam
    T_wc = np.linalg.inv(E_w2c)          # Cam->World
    T_wc_rot = rotate_Twc_around_z(pivot, T_wc, angle_rad)  # 你的函数：Cam->World
    E_w2c_rot = np.linalg.inv(T_wc_rot)  # 回到 World->Cam
    return E_w2c_rot

def transform_matrix_to_6d(Rts):
    """
    Convert [N, 4, 4] transformation matrices to [N, 6] pose vectors.
    Format: [rx, ry, rz, tx, ty, tz] where (rx, ry, rz) is the rotation vector.
    
    Args:
        Rts (np.ndarray): Shape [N, 4, 4]
        
    Returns:
        np.ndarray: Shape [N, 6]
    """
    # 提取旋转矩阵 [N, 3, 3]
    rot_matrices = Rts[:, :3, :3]
    # 提取平移向量 [N, 3]
    translations = Rts[:, :3, 3]
    
    # 将旋转矩阵转换为旋转向量 (Rotation Vector / Axis-Angle * angle)
    # scipy 的 as_rotvec() 返回的就是旋转向量
    rot_vectors = R.from_matrix(rot_matrices).as_rotvec()
    
    # 拼接 [N, 3] + [N, 3] -> [N, 6]
    poses_6d = np.concatenate([translations, rot_vectors], axis=1)
    
    return poses_6d

def read_data(pth):
    file_name = []
    with open(pth, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                file_name.append(line)
    return file_name

def _fill_grasp_trainval_infos(version="train", pruning=False):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    infos = []
    
    # filenames = sorted(os.listdir("data/grasp_anything/pc"))
    path = f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/"
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    filenames = sorted(folders, key=lambda s: int(s[:3]))

    print("Processing dataset for {} set!".format(version))

    id = 0 # glyou debug
    if version == "train":
        # train_data = read_data("/media/robot/data/WCL/taskgrasp/taskgrasp_image/splits_final/i/0/train_i.txt")
        # val_data  = read_data("/media/robot/data/WCL/taskgrasp/taskgrasp_image/splits_final/i/0/val_i.txt")
        # train_data = list(dict.fromkeys(train_data + val_data))
        # part_filenames = train_data
        part_filenames = filenames[id*seperate_num:min((id+1)*seperate_num, int(len(filenames)*4/5))]   # 80% scenes for training
    else:
        test_data = read_data("/media/robot/data/WCL/taskgrasp/taskgrasp_image/splits_final/i/0/test_i.txt")
        # part_filenames = test_data
        part_filenames = filenames[int(len(filenames)*4/5)+id*seperate_num:min(int(len(filenames)*4/5)+(id+1)*seperate_num, len(filenames))]   # 20% scenes for val
    


    # part_filenames = part_filenames[17:18]
    for jj, filename in enumerate(tqdm(part_filenames)):
        scene = filename
        # try: 
        #     with open(f"{root_path}/{scene}/.pkl", "rb") as f:
        #         prompts = pickle.load(f)
        # except:
        #     continue
        prompts = read_grasp_part(f"{path}/{scene}/visual_grasps/grasps_part.txt")

        # num_objects = len(prompts)
        gs_list = []
        gs_label_list = []

        pos_prompt_list = []


        pc = np.load(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{scene}/fused_pc_clean.npy")
        pc_mean = pc[:, :3].mean(axis=0)
        pc[:, :3] -= pc_mean
        z_min = pc[:, 2].min()
        eps = 1e-6  # 或者 1e-3 看你后续鲁棒性
        pc[:, 2] += (-z_min + eps)

        # with open(f"data/grasp_anything/grasp/{scene}_{i}", "rb") as f:
            # Rts, ws = pickle.load(f)
        Rts = merge_grasp(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{scene}")
        # gs = Rts.reshape(Rts.shape[0], -1)
        # gs = torch.from_numpy(gs).to(torch.float32)

        dz = -pc[:, 2].min() + eps  # 你对点云加的这个值
        T_shift = np.eye(4, dtype=np.float32)
        T_shift[2, 3] = dz
        Rts = T_shift[None, :, :] @ Rts

        grasp_pc = get_gripper_control_points()
        gs = np.matmul(Rts, grasp_pc.T).transpose(0, 2, 1)
        gs = gs[:, :, :3]
        gs = torch.from_numpy(gs).to(torch.float32)

        # gs = transform_matrix_to_6d(Rts)
        # ws = torch.zeros((len(Rts),), dtype=torch.float64).numpy()
        # gs = torch.from_numpy(np.concatenate((se3_log_map(torch.from_numpy(Rts)).numpy(), 2*ws[:, None]/MAX_WIDTH-1.0), axis=-1)).to(torch.float32)
        gs_labels = torch.ones_like(gs[..., :1], dtype=torch.int64)

        if pruning:
            num_grasps = len(gs)
            if num_grasps<=max_num:
                gs_list.append(gs)
                gs_label_list.append(gs_labels)
                # continue
            # assert gs.dim() == 2
            # pruned_grasps = pruning_grasps(gs, max_num=max_num)
            # gs_list.append(pruned_grasps)
            # gs_label_list.append(gs_labels[:len(pruned_grasps)])
        else:
            gs_list.append(gs)
            gs_label_list.append(gs_labels)

        pos_prompt_list.append(prompts)

        if len(gs_list)==0:
            continue



        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pc[:, :3])
        pivot = np.asarray(point_cloud.get_axis_aligned_bounding_box().get_center())
        
        pc_ori = torch.from_numpy(pc).to(torch.float32)
        pc_xyz = pc_ori[..., :3] # 0-1
        pc_rgb = pc_ori[..., 3:]/255.0 # 0-1
        
        n_view = 4
        fov_degree_list = [0, 90, 180, 270]
        
        theta_list = []
        for k in range(n_view):
            theta_list.append(np.radians(90 - fov_degree_list[k] / 2))

        out = np.array([[4.32978028e-17, -1.00000000e+00, 4.32978028e-17, -1.88532913e-02],
               [-7.07106781e-01, -6.12323400e-17, -7.07106781e-01, 2.81969264e-03],
               [7.07106781e-01, -4.35788200e-33, -7.07106781e-01, 4.05902828e-01],
               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # ====== 动态计算 out ======
        angle_z = np.radians(-90)
        angle_y = np.radians(-45)

        Rz = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z),  np.cos(angle_z), 0],
            [0,                0,               1],
        ], dtype=np.float64)

        Ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)],
        ], dtype=np.float64)

        R_world = Ry @ Rz

        # 构造 T_wc
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :3] = R_world

        # 确定相机位置
        # 根据包围盒大小自适应距离，确保物体在视野内
        bbox = point_cloud.get_axis_aligned_bounding_box()
        extent = np.linalg.norm(np.asarray(bbox.get_extent()))
        dist = extent * 1.5
        # dist = max(extent * 2.0, 0.8) # 经验值，至少0.8米

        # 相机位置 = pivot + Z轴方向 * 距离
        # 确保相机看向 pivot (相机 Z 轴指向后方，所以放在 +Z 处)
        z_axis = T_wc[:3, 2] 
        T_wc[:3, 3] = pivot + z_axis * dist

        out = np.linalg.inv(T_wc)
        # ==========================


        depth_map_list, ext_list, K_list = [], [], []
        for k in range(n_view):

            K = torch.tensor([[665.10751011, 0, 511.5, 0],
                            [0, 665.10751011, 383.5, 0],
                            [0, 0,  1, 0]], dtype=torch.float32)
            angle = np.radians(fov_degree_list[k])
            # ext = rotate_Twc_around_z(pivot, out, angle)
            ext_np = rotate_extrinsic_world2cam_around_z(pivot, out, angle)
            ext = torch.from_numpy(ext_np).to(torch.float32)
            
            proj_points = project_points(pc_xyz, K, ext)

            # Normalize to 0-1 (assuming original resolution 1024x768 based on K)
            proj_points[:, 0] = proj_points[:, 0] / 1024.0
            proj_points[:, 1] = proj_points[:, 1] / 768.0
            
            depth_map, img_rgb = points2deprgb(proj_points, pc_rgb, width, height)

            # print(f"Depth min: {depth_map.min()}, max: {depth_map.max()}")
            # depth = depth_map.detach().cpu().numpy().astype(np.float32)
            # valid = np.isfinite(depth) & (depth > 0)   # 常见：0 表示无效
            # vmin, vmax = np.percentile(depth[valid], [2, 98])  # 比 min/max 更稳
            # depth_clip = np.clip(depth, vmin, vmax)
            # depth_norm = ((depth_clip - vmin) / (vmax - vmin + 1e-6) * 255).astype(np.uint8)
            # depth_norm[~valid] = 0
            # heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            # cv2.imwrite(
            #     f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{scene}/rgb/{scene}_{str(k)}_depth_heatmap.png",
            #     heatmap
            # )

            img_rgb = img_rgb.detach().cpu().numpy()


            save_rgb_dir = f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{scene}/rgb/"
            os.makedirs(save_rgb_dir, exist_ok=True)

            # if not os.path.exists(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{scene}/rgb/{scene}_{str(k)}.png"):
            cv2.imwrite(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{scene}/rgb/{scene}_{str(k)}.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            
            # img_rgb_list.append(img_rgb) # 0-255
            depth_map_list.append(depth_map) # 0-255
            ext_list.append(torch.linalg.inv(ext))
            K_list.append(K)
        # img_rgb_all = torch.stack(img_rgb_list).numpy()
        depth_map_all = torch.stack(depth_map_list).numpy()
        # print("depth_map_all",depth_map_all)
        ext_all = torch.stack(ext_list).to(torch.float32).numpy()
        K_all =  torch.stack(K_list).to(torch.float32).numpy()


        save_depth_dir = f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{scene}/depth/"
        os.makedirs(save_depth_dir, exist_ok=True)

        np.save(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{scene}/depth/{scene}.npy", depth_map_all)

        info = {
            'scene_token': scene,
            'gs_prompts': pos_prompt_list,
            'gs': gs_list,
            'gs_labels': gs_label_list,
            'pc_path': filename,
            'img_path': f"{scene}/rgb/{scene}",
            'depth_path': f"{scene}/depth/{scene}",
            'pose': ext_all,
            'intrinsic': K_all,
        }
        infos.append(info)

    return infos


def create_grasp_infos(root_path,
                          info_prefix,
                          version='train',
                          pruning=True,
                          id=0):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """

    infos = _fill_grasp_trainval_infos(version=version, pruning=pruning)

    metadata = dict(version=version)

    print('{} sample: {}'.format(version, len(infos)))
    data = dict(infos=infos, metadata=metadata)
    info_path = osp.join(root_path,
                            '{}_infos_{}_'.format(info_prefix, version)+str(id)+'.pkl')
    dump(data, info_path)
    print('Finish {}_infos_{}_'.format(info_prefix, version)+str(id))



def parse_args():
    parser = argparse.ArgumentParser(description="Create grasp data! ")
    # parser.add_argument('--version', required=True)
    # parser.add_argument('--pruning', action="store_true")
    # parser.add_argument("--id", type=int, help="dataset is too big and needs ids to seperate.")
    parser.add_argument('--version', default="test")
    parser.add_argument('--pruning', default=True)
    parser.add_argument("--id", type=int, default=0)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    create_grasp_infos("/media/robot/data/WCL/taskgrasp/taskgrasp_image/",
                          "grasp_task",
                          version=args.version,
                          pruning=args.pruning,
                          id=args.id)
