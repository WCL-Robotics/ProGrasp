import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
import imageio
from skimage.transform import rotate, resize
import open3d as o3d
import trimesh.transformations as tra
from sklearn.decomposition import PCA
import os
import torch


def get_gripper_control_points():
    return np.array([
        [-0.10, 0, 0, 1],
        [-0.03, 0, 0, 1],
        [-0.03, 0.07, 0, 1],
        [0.03, 0.07, 0, 1],
        [-0.03, -0.07, 0, 1],
        [0.03, -0.07, 0, 1]])

def depth_to_pointcloud(
        depth: np.ndarray,
        K: np.ndarray,
        depth_scale: float = 1.0,
        depth_trunc: float = None,
        validity_mask: np.ndarray = None,
        return_mask=False
    ) -> np.ndarray:
    """
    将深度图转换为 Nx3 点云（右手坐标系，Z 轴朝前）。

    参数
    ----
    depth : (H, W) float32/float64
        深度图，单位与 depth_scale 一致（如 1 表示 1 米、或 1000 表示 1 毫米）。
    K : (3, 3) ndarray
        内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]。
    depth_scale : float
        深度缩放系数。若深度图单位为毫米，可设为 1000，使得结果以米为单位。
    depth_trunc : float | None
        深度上限（米）。大于该值的像素将被丢弃。None 表示忽略。
    validity_mask : (H, W) bool | None
        额外的有效性掩码；True 表示保留，False 表示丢弃。

    返回
    ----
    points : (N, 3) ndarray
        点云坐标 (X, Y, Z)，单位与 depth_scale 相同。
    """
    assert depth.ndim == 2, "depth 应为单通道图"
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 生成像素网格（u,v）
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # 转为米（或任意目标单位）
    z = depth.astype(np.float64) / depth_scale
    if depth_trunc is not None:
        mask = (z > 0) & (z < depth_trunc)
    else:
        mask = z > 0
    if validity_mask is not None:
        mask &= validity_mask

    z = z[mask]
    u = u[mask]
    v = v[mask]

    # 反投影
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.column_stack((x, y, z))  # (N,3)
    if return_mask:
        return points, mask
    return points

def get_gripper_control_points():
    return np.array([
        [-0.10, 0, 0, 1],
        [-0.03, 0, 0, 1],
        [-0.03, 0.07, 0, 1],
        [0.03, 0.07, 0, 1],
        [-0.03, 0.07, 0, 1],
        [-0.03, -0.07, 0, 1],
        [0.03, -0.07, 0, 1]])

def get_gripper_control_points_o3d(
    grasp,
    direction_vector=False,
    show_sweep_volume=False,
    color=(
        0.2,
        0.8,
        0)):
    """
    Open3D Visualization of parallel-jaw grasp

    grasp: [4, 4] np array
    """

    meshes = []
    align = tra.euler_matrix(np.pi / 2, 0, 0)
    if direction_vector:
        # ---------- 先算两根杆的中心位置 ----------
        # 中间长杆 cylinder_1 的变换
        T1_local = np.eye(4)
        T1_local[0, 3] = -0.03
        T1 = grasp @ (align @ T1_local)
        center_1_world = T1[:3, 3]

        # 横向杆 cylinder_2 的变换
        T2_local = tra.euler_matrix(0, np.pi / 2, 0)
        T2_local[0, 3] = -0.065
        T2 = grasp @ (align @ T2_local)
        center_2_world = T2[:3, 3]

        # 从中间长杆指向横向杆的方向向量（世界坐标系）
        dir_1_to_2 = center_2_world - center_1_world
        dir_1_to_2 = dir_1_to_2 / np.linalg.norm(dir_1_to_2)
        # print("Direction from middle bar to horizontal bar:", dir_1_to_2)

    # Cylinder 3,5,6
    cylinder_1 = o3d.geometry.TriangleMesh.create_cylinder(
        radius=0.005, height=0.139)
    transform = np.eye(4)
    transform[0, 3] = -0.03
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    cylinder_1.paint_uniform_color(color)
    cylinder_1.transform(transform)

    # Cylinder 1 and 2
    cylinder_2 = o3d.geometry.TriangleMesh.create_cylinder(
        radius=0.005, height=0.07)
    transform = tra.euler_matrix(0, np.pi / 2, 0)
    transform[0, 3] = -0.065
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    cylinder_2.paint_uniform_color(color)
    cylinder_2.transform(transform)

    # Cylinder 5,4
    cylinder_3 = o3d.geometry.TriangleMesh.create_cylinder(
        radius=0.005, height=0.06)
    transform = tra.euler_matrix(0, np.pi / 2, 0)
    transform[2, 3] = 0.065
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    cylinder_3.paint_uniform_color(color)
    cylinder_3.transform(transform)

    # Cylinder 6, 7
    cylinder_4 = o3d.geometry.TriangleMesh.create_cylinder(
        radius=0.005, height=0.06)
    transform = tra.euler_matrix(0, np.pi / 2, 0)
    transform[2, 3] = -0.065
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    cylinder_4.paint_uniform_color(color)
    cylinder_4.transform(transform)

    cylinder_1.compute_vertex_normals()
    cylinder_2.compute_vertex_normals()
    cylinder_3.compute_vertex_normals()
    cylinder_4.compute_vertex_normals()

    meshes.append(cylinder_1)
    meshes.append(cylinder_2)
    meshes.append(cylinder_3)
    meshes.append(cylinder_4)

    # Just for visualizing - sweep volume
    if show_sweep_volume:
        finger_sweep_volume = o3d.geometry.TriangleMesh.create_box(
            width=0.06, height=0.02, depth=0.14)
        transform = np.eye(4)
        transform[0, 3] = -0.06 / 2
        transform[1, 3] = -0.02 / 2
        transform[2, 3] = -0.14 / 2

        transform = np.matmul(align, transform)
        transform = np.matmul(grasp, transform)
        finger_sweep_volume.paint_uniform_color([0, 0.2, 0.8])
        finger_sweep_volume.transform(transform)
        finger_sweep_volume.compute_vertex_normals()

        meshes.append(finger_sweep_volume)
    if direction_vector:
        return meshes, dir_1_to_2
    return meshes

def get_cube_mesh(
    pose,
    color=(1, 0.84, 0), # 金色
    size=0.01):
    """
    创建一个立方体 Mesh 用于可视化。
    """
    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    
    # Center the cube
    mesh.translate([-size/2, -size/2, -size/2])
    
    # 只应用平移，忽略旋转
    translation = pose[:3, 3]
    mesh.translate(translation)
    
    return [mesh]

def get_gripper_arrow_mesh_o3d(
    grasp: np.ndarray,
    color=(0.9, 0.2, 0.2),
    cylinder_radius=0.006,
    cone_radius=0.015,
    cylinder_height=0.08,
    cone_height=0.03,
):
    """
    Create an Open3D arrow mesh for the gripper.

    Requirements satisfied:
    1) Arrow is parallel to cylinder_3 and cylinder_4 (same rotation chain).
    2) Arrow lies exactly between cylinder_3 and cylinder_4 (symmetric midpoint, z-offset = 0).
    3) Arrow tip is exactly at grasp translation (grasp[:3,3]).

    grasp: (4,4) homogeneous transform
    """

    grasp = np.asarray(grasp, dtype=np.float64)
    assert grasp.shape == (4, 4)

    # Same "align" as your gripper visualization
    align = tra.euler_matrix(np.pi / 2, 0, 0)

    # Same local rotation used by cylinder_3 / cylinder_4
    # (they differ only by transform[2,3]=+/-0.065, so midpoint uses 0 offset)
    R_local = tra.euler_matrix(0, np.pi / 2, 0)

    # Open3D create_arrow is along +Z; its tip is at z = cylinder_height + cone_height
    total_len = float(cylinder_height + cone_height)

    # Shift arrow so that its TIP is at the local origin before applying grasp
    tip_to_origin = np.eye(4)
    tip_to_origin[2, 3] = -total_len

    # Final transform: grasp @ (align @ R_local @ tip_to_origin)
    T = grasp @ (align @ (R_local @ tip_to_origin))

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=cylinder_height,
        cone_height=cone_height,
    )
    arrow.paint_uniform_color(color)
    arrow.transform(T)
    arrow.compute_vertex_normals()

    return [arrow]



def pca_normalise(points, scale_mode='none'):
    """
    使用 PCA 将点云规范化：
    ① 平移到原点；② 旋转对齐主成分；③ 按需缩放。

    参数
    ----
    points : ndarray, shape (N, 3)
        输入点云
    scale_mode : {'none', 'unit_cube', 'unit_sphere'}
        - 'none'        ：仅平移+旋转
        - 'unit_cube'   ：按最大轴向范围缩放到 [-1,1] 立方体
        - 'unit_sphere' ：按最大半径缩放到单位球（默认）

    返回
    ----
    normed_pts : (N, 3) ndarray   规范化后的点云
    R          : (3, 3) ndarray   旋转矩阵（行向量为新坐标轴在原坐标系中的表示）
    centroid   : (3,)   ndarray   原始质心
    """
    pts = np.asarray(points, dtype=float)
    centroid = pts.mean(axis=0)           # ① 质心
    pts_c = pts - centroid                # ② 平移至原点

    pca = PCA(n_components=3)
    pts_r = pca.fit_transform(pts_c)      # ③ 旋转到主轴

    if scale_mode == 'unit_cube':
        scale = np.abs(pts_r).max()
        pts_r /= scale
    elif scale_mode == 'unit_sphere':
        scale = np.linalg.norm(pts_r, axis=1).max()
        pts_r /= scale
    # 若 scale_mode == 'none' 则不缩放

    return pts_r, pca.components_, centroid

def depth_to_pcd(camera_path, depth_path, rgb_path, mask_path=None):
    depth_data = np.load(depth_path)
    camera_info = np.load(camera_path)
    validity_mask = None
    if mask_path is not None:
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        validity_mask = mask_img > 0  # 白色为True，黑色为False

    point_cloud, mask = depth_to_pointcloud(
        depth_data,
        camera_info,
        depth_scale=1000.0,
        depth_trunc=None,           # 丢弃 2m 以外的点（可选）
        validity_mask=validity_mask,
        return_mask=True
    )
    rgb_img = cv2.imread(rgb_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    colors = rgb_img[mask]

    pcd = o3d.geometry.PointCloud()
    # 设置点云坐标 (前3列)
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    # 3. 设置点云颜色 (需要归一化到 [0, 1])
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    return pcd

def merge_grasp(grasp_path):
    all_grasps = []
    for i in range(25):  # 0-24
        grasp_file = f"{grasp_path}/grasps/{i}/grasp.npy"
        try:
            grasp_data = np.load(grasp_file)
            all_grasps.append(grasp_data)

        except FileNotFoundError:
            print(f"Warning: {grasp_file} not found, skipping...")
        except Exception as e:
            print(f"Error loading {grasp_file}: {e}")
    
    # 将所有抓取姿态合并为一个数组
    if all_grasps:
        grasps = np.array(all_grasps)
        print(f"Total aggregated grasps shape: {grasps.shape}")
        return grasps
    else:
        print("No grasp files found!")
        return


def farthest_points(
        data,
        nclusters,
        dist_func,
        return_center_indexes=False,
        return_distances=False,
        verbose=False):
    """
      Code taken from https://github.com/NVlabs/6dof-graspnet

      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of
          clusters.
        return_distances: bool, If True, return distances of each point from centers.

      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(
                data.shape[0], dtype=np.int32), np.arange(
                data.shape[0], dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0],), dtype=np.int32) * -1
    distances = np.ones((data.shape[0],), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print(
                'farthest points max distance : {}'.format(
                    np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters


def distance_by_translation_grasp(p1, p2):
    """
      Gets two nx4x4 numpy arrays and computes the translation of all the
      grasps.
    """
    t1 = p1[:, :3, 3]
    t2 = p2[:, :3, 3]
    return np.sqrt(np.sum(np.square(t1 - t2), axis=-1))

def cluster_grasps(grasps, num_clusters=32):
    ratio_of_grasps_to_be_used = 1.0
    cluster_indexes = np.asarray(
        farthest_points(
            grasps,
            num_clusters,
            distance_by_translation_grasp))
    output_grasps = []

    for i in range(num_clusters):
        indexes = np.where(cluster_indexes == i)[0]
        if ratio_of_grasps_to_be_used < 1:
            num_grasps_to_choose = max(
                1, int(ratio_of_grasps_to_be_used * float(len(indexes))))
            if len(indexes) == 0:
                raise ValueError('Error in clustering grasps')
            indexes = np.random.choice(
                indexes, size=num_grasps_to_choose, replace=False)

        output_grasps.append(grasps[indexes, :, :])

    # output_grasps = np.asarray(output_grasps)
    output_grasps = np.asarray(output_grasps)

    return output_grasps


def sample_grasp_indexes(n, grasps):
    """
        Stratified sampling of the graps.
    """
    nonzero_rows = [i for i in range(len(grasps)) if len(grasps[i]) > 0]
    num_clusters = len(nonzero_rows)
    replace = n > num_clusters
    assert num_clusters != 0

    grasp_rows = np.random.choice(
        range(num_clusters),
        size=n,
        replace=replace).astype(
        np.int32)
    grasp_rows = [nonzero_rows[i] for i in grasp_rows]
    grasp_cols = []
    for grasp_row in grasp_rows:
        if len(grasps[grasp_rows]) == 0:
            raise ValueError('grasps cannot be empty')

        grasp_cols.append(np.random.randint(len(grasps[grasp_row])))

    grasp_cols = np.asarray(grasp_cols, dtype=np.int32)

    return np.vstack((grasp_rows, grasp_cols)).T

def farthest_grasps(grasps, num_clusters=32, num_grasps=64):
    """ Returns grasps sampled with farthest point sampling """
    grasps_fps = cluster_grasps(grasps, num_clusters=num_clusters)
    clusters_fps = sample_grasp_indexes(num_grasps, grasps_fps)
    grasps = np.array([grasps_fps[cluster[0]][cluster[1]]
                       for cluster in clusters_fps])
    return grasps

def distance_by_translation_point(p1, p2):
    """
      Gets two nx3 points and computes the disntace between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))

def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates whether to use farthest point sampling
      to downsample the points. Farthest point sampling version runs slower.
    """
    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(
                pc, npoints, distance_by_translation_point, return_center_indexes=True)
        else:
            center_indexes = np.random.choice(
                range(pc.shape[0]), size=npoints, replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc

def pc_to_depth(xyz: np.ndarray, cam_info: np.ndarray, height: int, width: int):
    uvd = xyz @ cam_info.T
    uvd /= uvd[:, 2:]
    uv = uvd[:, :2].astype(np.int32)

    img = np.zeros((height, width), dtype=np.float32)
    mask = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv = uv[mask]
    xyz = xyz[mask]
    img[uv[:, 1], uv[:, 0]] = xyz[:, 2]
    return img

def show_grasps_sequential(point_cloud, coordinate_frame, grasps, save_dir=None):
    vis = o3d.visualization.Visualizer()  # 不再需要 KeyCallback
    vis.create_window(window_name="Grasp Viewer", width=1024, height=768)

    vis.add_geometry(point_cloud)
    # vis.add_geometry(coordinate_frame)

    # 让 Open3D 用 bbox 初始化一次相机
    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()

    ctr = vis.get_view_control()
    cam = ctr.convert_to_pinhole_camera_parameters()


    pivot = np.asarray(point_cloud.get_axis_aligned_bounding_box().get_center())

    # ====== 主视图的旋转设置 ======
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

    # 一次性把相机绕 pivot 转到你想要的角度
    E = cam.extrinsic.astype(np.float64)   # World->Cam
    T_wc = np.linalg.inv(E)                # Cam->World
    C = T_wc[:3, 3]                        # 相机在世界坐标系下的位置

    # 让相机离物体远一点
    dist_scale = 1.5
    v = C - pivot
    v_rot = R_world @ v
    v_rot_far = dist_scale * v_rot

    T_wc[:3, :3] = R_world @ T_wc[:3, :3]
    T_wc[:3, 3]  = pivot + v_rot_far

    # 把“主视图”的相机位姿存下来（Cam->World）
    T_wc_main = T_wc.copy()


    try:
        ctr.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
    except TypeError:
        ctr.convert_from_pinhole_camera_parameters(cam)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # === 辅助函数：给定 T_wc 设置相机、渲染并截图 ===
    def capture_view(T_wc_view, filename):
        cam_local = ctr.convert_to_pinhole_camera_parameters()
        cam_local.extrinsic = np.linalg.inv(T_wc_view)

        try:
            ctr.convert_from_pinhole_camera_parameters(cam_local, allow_arbitrary=True)
        except TypeError:
            ctr.convert_from_pinhole_camera_parameters(cam_local)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(filename, do_render=False)
        print(f"Saved image to: {filename}")

    # === 辅助函数：绕 pivot 的 z 轴旋转相机（Cam->World） ===
    def rotate_Twc_around_z(T_wc_base, angle_rad):
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

    # === 核心：不再交互，而是直接遍历所有 grasps ===
    grasp_geoms = []
    direction_vector = []
    for i in range(len(grasps)):
        # 移除旧抓取
        for g in grasp_geoms:
            vis.remove_geometry(g, reset_bounding_box=False)

        # 添加新抓取
        grasp_geoms, dir = get_gripper_control_points_o3d(grasps[i],direction_vector=True)
        # grasp_geoms = get_cube_mesh(grasps[i])
        # grasp_geoms = get_gripper_arrow_mesh_o3d(grasps[i])
        # direction_vector.append(dir)
        
        for g in grasp_geoms:
            vis.add_geometry(g, reset_bounding_box=False)

        vis.poll_events()
        vis.update_renderer()

        print(f"\nProcessing grasp {i+1}/{len(grasps)}")

        if save_dir is not None:
            # 每个 grasp 单独一个文件夹：save_dir/i/
            main_folder = os.path.join(save_dir, f"{i}")
            os.makedirs(main_folder, exist_ok=True)

            # 主视图
            main_path = os.path.join(main_folder, "grasp_main.png")
            capture_view(T_wc_main, main_path)            

            # 右视图
            angle_right = np.radians(60.0)
            T_wc_right = rotate_Twc_around_z(T_wc_main, angle_right)
            right_path = os.path.join(main_folder, "grasp_right.png")
            capture_view(T_wc_right, right_path)

            # 左视图
            angle_left = np.radians(-60.0)
            T_wc_left = rotate_Twc_around_z(T_wc_main, angle_left)
            left_path = os.path.join(main_folder, "grasp_left.png")
            capture_view(T_wc_left, left_path)

            # 恢复相机到主视图，方便下一轮循环
            cam_restore = ctr.convert_to_pinhole_camera_parameters()
            cam_restore.extrinsic = np.linalg.inv(T_wc_main)
            try:
                ctr.convert_from_pinhole_camera_parameters(cam_restore, allow_arbitrary=True)
            except TypeError:
                ctr.convert_from_pinhole_camera_parameters(cam_restore)
    
    # if save_dir is not None:
    #     direction_vector = np.array(direction_vector)
    #     np.save(os.path.join(save_dir, "direction_vector.npy"), direction_vector)
    vis.destroy_window()

def show_point_sequential(pc_list, coordinate_frame, grasps, save_dir=None):
    vis = o3d.visualization.Visualizer()  # 不再需要 KeyCallback
    vis.create_window(window_name="Grasp Viewer", width=1024, height=768)

    pc_input = pc_list[0]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc_input[:, :3])
    point_cloud.colors = o3d.utility.Vector3dVector(pc_input[:, 3:]/255.0)
    grasp_geoms, dir = get_gripper_control_points_o3d(grasps[0],direction_vector=True)
    
    
    
    vis.add_geometry(coordinate_frame)

    # 让 Open3D 用 bbox 初始化一次相机
    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()

    ctr = vis.get_view_control()
    cam = ctr.convert_to_pinhole_camera_parameters()

    # 使用点云的包围盒中心作为 pivot
    bbox = grasp_geoms[0].get_axis_aligned_bounding_box()
    pivot = np.asarray(bbox.get_center())
    extent = np.linalg.norm(bbox.get_extent())

    # ====== 主视图的旋转设置 ======
    angle_z = np.radians(-60)
    angle_y = np.radians(30)

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

    # Calculate front vector (direction from Pivot to Camera)
    # Initial front (before rotation) assumed to be +Z direction
    v_front = R_world @ np.array([0, 0, 1], dtype=np.float64)

    # Apply view settings
    ctr.set_lookat(pivot)
    ctr.set_front(v_front)
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(2) # Adjust zoom as needed

    vis.poll_events()
    vis.update_renderer()

    # Capture the resulting camera pose
    cam = ctr.convert_to_pinhole_camera_parameters()
    T_wc_main = np.linalg.inv(cam.extrinsic)
    camera_offset = T_wc_main[:3, 3] - pivot

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # === 辅助函数：给定 T_wc 设置相机、渲染并截图 ===
    def capture_view(T_wc_view, filename):
        cam_local = ctr.convert_to_pinhole_camera_parameters()
        cam_local.extrinsic = np.linalg.inv(T_wc_view)

        try:
            ctr.convert_from_pinhole_camera_parameters(cam_local, allow_arbitrary=True)
        except TypeError:
            ctr.convert_from_pinhole_camera_parameters(cam_local)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(filename, do_render=False)
        print(f"Saved image to: {filename}")

    # === 辅助函数：绕 pivot 的 z 轴旋转相机（Cam->World） ===
    def rotate_Twc_around_z(T_wc_base, angle_rad):
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

    # === 核心：不再交互，而是直接遍历所有 grasps ===
    grasp_geoms = []
    direction_vector = []
    pc_geoms = []
    for i in range(len(grasps)):
        # 移除旧抓取
        # for g in grasp_geoms:
        #     vis.remove_geometry(g, reset_bounding_box=False)
        for p in pc_geoms:
            vis.remove_geometry(p, reset_bounding_box=False)

        # 添加新抓取
        grasp_geoms, dir = get_gripper_control_points_o3d(grasps[i],direction_vector=True)
        # grasp_geoms = get_cube_mesh(grasps[i])
        # grasp_geoms = get_gripper_arrow_mesh_o3d(grasps[i])
        # direction_vector.append(dir)
        pc_input = pc_list[i]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pc_input[:, :3])
        point_cloud.colors = o3d.utility.Vector3dVector(pc_input[:, 3:]/255.0)
        pc_geoms = [point_cloud]
        
        # for g in grasp_geoms:
        #     vis.add_geometry(g, reset_bounding_box=False)
        for p in pc_geoms:
            vis.add_geometry(p, reset_bounding_box=False)

        # Update camera to look at the new point cloud center
        bbox = grasp_geoms[0].get_axis_aligned_bounding_box()
        pivot = np.asarray(bbox.get_center())
        
        # Update T_wc_current to follow the object
        T_wc_current = T_wc_main.copy()
        T_wc_current[:3, 3] = pivot + camera_offset
        
        # Update the view control to this new pose
        cam_current = ctr.convert_to_pinhole_camera_parameters()
        cam_current.extrinsic = np.linalg.inv(T_wc_current)
        try:
            ctr.convert_from_pinhole_camera_parameters(cam_current, allow_arbitrary=True)
        except TypeError:
            ctr.convert_from_pinhole_camera_parameters(cam_current)

        vis.poll_events()
        vis.update_renderer()

        print(f"\nProcessing grasp {i+1}/{len(grasps)}")

        if save_dir is not None:
            # 每个 grasp 单独一个文件夹：save_dir/i/
            main_folder = os.path.join(save_dir, f"{i}")
            os.makedirs(main_folder, exist_ok=True)

            # 主视图
            main_path = os.path.join(main_folder, "grasp_main.png")
            capture_view(T_wc_current, main_path)            

            # 右视图
            angle_right = np.radians(60.0)
            T_wc_right = rotate_Twc_around_z(T_wc_current, angle_right)
            right_path = os.path.join(main_folder, "grasp_right.png")
            capture_view(T_wc_right, right_path)

            # 左视图
            angle_left = np.radians(-60.0)
            T_wc_left = rotate_Twc_around_z(T_wc_current, angle_left)
            left_path = os.path.join(main_folder, "grasp_left.png")
            capture_view(T_wc_left, left_path)

            # 恢复相机到主视图，方便下一轮循环
            cam_restore = ctr.convert_to_pinhole_camera_parameters()
            cam_restore.extrinsic = np.linalg.inv(T_wc_current)
            try:
                ctr.convert_from_pinhole_camera_parameters(cam_restore, allow_arbitrary=True)
            except TypeError:
                ctr.convert_from_pinhole_camera_parameters(cam_restore)
    
    # if save_dir is not None:
    #     direction_vector = np.array(direction_vector)
    #     np.save(os.path.join(save_dir, "direction_vector.npy"), direction_vector)
    vis.destroy_window()



def visualize_pc_data():
    all_grasp_meshes = []

    # object = "008_masher"
    # trans_path = f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{object}/0_pc_to_img_trf.npy"
    # trans = np.load(trans_path)

    # grasp_path = f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{object}"
    # grasps = merge_grasp(grasp_path)
    # grasps = trans @ grasps

    # camera_path = f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{object}/0_camerainfo.npy"
    # depth_path = f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{object}/0_depth.npy"
    # rgb_path = f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{object}/0_color.png"
    # mask_path = f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{object}/0_color_mask.png"
    # depth_data = np.load(depth_path)
    # camera_info = np.load(camera_path)
    # point_cloud, mask = depth_to_pointcloud(
    #     depth_data,
    #     camera_info,
    #     depth_scale=1000.0,
    #     depth_trunc=None,           # 丢弃 2m 以外的点（可选）
    #     validity_mask=None,
    #     return_mask=True
    # )
    # point_cloud[:, 0] += 0.021
    # point_cloud[:, 1] -= 0.003
    # rgb_img = cv2.imread(rgb_path)
    # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    # corr_depth = 1000*pc_to_depth(point_cloud[:,:3], camera_info, rgb_img.shape[0], rgb_img.shape[1])

    # # --- Project Grasps to 2D Image ---
    # for index in range(len(grasps)):
    #     # 1. Extract XYZ from grasps (N, 4, 4) -> (N, 3)
    #     grasp_xyz = grasps[:, :3, 3].copy()
    #     grasp_xyz = grasp_xyz[index:index+1]
    #     print(index)

    #     # 2. Apply the same offset as point_cloud to align with RGB
    #     grasp_xyz[:, 0] += 0.021
    #     grasp_xyz[:, 1] -= 0.002

    #     # 3. Project to 2D pixel coordinates
    #     # uvw = xyz @ K.T
    #     uvw = grasp_xyz @ camera_info.T
    #     uv = uvw[:, :2] / uvw[:, 2:]  # Divide by Z to get (u, v)
    #     uv = uv.astype(np.int32)

    #     # 4. Draw yellow squares on the image
    #     img_vis = rgb_img.copy()
    #     square_size = 10
    #     half_size = square_size // 2
    #     yellow_color = (255, 255, 0)  # RGB for Yellow

    #     for point in uv:
    #         u, v = point[0], point[1]
    #         # Check if point is within image bounds
    #         if 0 <= u < img_vis.shape[1] and 0 <= v < img_vis.shape[0]:
    #             top_left = (u - half_size, v - half_size)
    #             bottom_right = (u + half_size, v + half_size)
    #             cv2.rectangle(img_vis, top_left, bottom_right, yellow_color, 2)

    #     # 5. Save or Display
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(img_vis)
    #     plt.title("Grasp Projections")
    #     plt.axis('off')
    #     plt.show()

    

    # mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # validity_mask = mask_img > 0  # 白色为True，黑色为False
    # pcd, mask = depth_to_pointcloud(
    #     corr_depth,
    #     camera_info,
    #     depth_scale=1000.0,
    #     depth_trunc=None,           # 丢弃 2m 以外的点（可选）
    #     validity_mask=validity_mask,
    #     return_mask=True
    # )
    # pcd = np.load(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{object}/0_segmented_pc.npy")
    # mean = pcd[:, :3].mean(axis=0)
    # pcd, mask = depth_to_pointcloud(
    #     corr_depth,
    #     camera_info,
    #     depth_scale=1000.0,
    #     depth_trunc=None,           # 丢弃 2m 以外的点（可选）
    #     validity_mask=None,
    #     return_mask=True
    # )
    # rgb_img = cv2.imread(rgb_path)
    # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    # colors = rgb_img[mask]


    # pcd[:, :3] -= mean
    # grasps[:, :3, 3] -= mean
    

    # for i in range(0, len(grasps)):
    #     # grasp_mesh = get_gripper_control_points_o3d(grasps[i])
    #     grasp_mesh = get_cube_mesh(grasps[i])
    #     all_grasp_meshes.extend(grasp_mesh)
    

    # pointcloud = o3d.geometry.PointCloud()
    # pointcloud.points = o3d.utility.Vector3dVector(pcd[:, :3])
    # pointcloud.colors = o3d.utility.Vector3dVector(pcd[:, 3:]/255.0)

    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=0.1,  # 坐标轴长度
    #     origin=[0, 0, 0]  # 原点位置
    # )

    # o3d.visualization.draw_geometries([pointcloud, coordinate_frame] + all_grasp_meshes[6:7], 
    #                                 window_name="Point Cloud with RGB Colors and Origin",
    #                                 width=1024, 
    #                                 height=768,
    #                                 point_show_normal=False)

    # ---- 主统计 ----
    global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)


    task_obj = '/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans'
    folders = [f for f in os.listdir(task_obj) if os.path.isdir(os.path.join(task_obj, f))]
    folders_sorted = sorted(folders, key=lambda s: int(s[:3]))
    folders_sorted_selected = folders_sorted[139:140]

    for file in folders_sorted_selected:
        grasp_path = f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}"
        grasps = merge_grasp(grasp_path)

        pcd = np.load(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/fused_pc_clean.npy")
        # Downsample to 8192 points
        if pcd.shape[0] > 4096:
            _, indices = farthest_points(pcd[:, :3], 4096, distance_by_translation_point, return_center_indexes=True)
            pcd = pcd[indices]

        # pc_mean = pcd[:, :3].mean(axis=0)
        # pcd[:, :3] -= pc_mean
        # z_min = pcd[:, 2].min()
        # eps = 1e-6  # 或者 1e-3 看你后续鲁棒性
        # pcd[:, 2] += (-z_min + eps)
        # pcd = pcd[:, :3]

        # # 更新全局 min/max
        # mn = pcd.min(axis=0)
        # mx = pcd.max(axis=0)
        # global_min = np.minimum(global_min, mn)
        # global_max = np.maximum(global_max, mx)

        # np.save(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/down_pc_4096.npy", pcd)
        # print("save downsampled point cloud for", file, "with shape:", pcd.shape)

        pc_mean = pcd[:, :3].mean(axis=0)
        pcd[:, :3] -= pc_mean
        z_min = pcd[:, 2].min()
        eps = 1e-6  # 或者 1e-3 看你后续鲁棒性
        pcd[:, 2] += (-z_min + eps)

        dz = -z_min + eps  # 你对点云加的这个值
        T_shift = np.eye(4, dtype=np.float32)
        T_shift[2, 3] = dz
        grasps = T_shift[None, :, :] @ grasps

        pc_input = pcd.copy()

        level_grasp = np.array(
            [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
        )
        vertical_grasp = tra.euler_matrix(0, np.pi / 2, 0) 
        
        # 2. 平移：设置一个便于观察的位置 (例如在物体上方 10cm 处)
        vertical_grasp[:3, 3] = [0, 0, 0]

        trans_matrices = vertical_grasp @ np.linalg.inv(grasps)
        # trans_matrices = trans_matrices[0]
        pc_list = []
        for index in range(len(trans_matrices)):
            temp = trans_matrices[index]
            xyz_transformed = pc_input[:, :3] @ temp[:3, :3].T + temp[:3, 3]
            pc_transformed = np.concatenate([xyz_transformed, pc_input[:, 3:]], axis=1)
            pc_list.append(pc_transformed)

        vertical_grasp_expanded = np.tile(vertical_grasp[np.newaxis, :, :], (len(grasps), 1, 1))

        grasp_pc = get_gripper_control_points()
        grasp_pc = np.matmul(grasps, grasp_pc.T).transpose(0, 2, 1)
        grasp_pc = grasp_pc[:, :, :3]

        
        # part-related：152_spoon [2, 12, 14, 18, 19]
        # 020_mug [1, 2, 7, 8 ,9]
        # 178_fork [11, 17, 21, 23]
        # part-irrelevant：158_spoon [5, 0, 8, 21, 14] 135_hammer [0, 3, 5, 10, 11] 229_screwdriver
        # Category-related 174_spatula [3, 15, 7, 8, 9]
        # 160_brush [2, 4, 7, 21]
        # 114_scissors [0, 1, 4, 5]
        num = [0, 3, 5, 10, 11]
        
                                
        # for i in range(19, len(grasps)):
        for i in num:
            grasp_mesh = get_gripper_control_points_o3d(grasps[i], color=(1.0, 0.0, 0.0))
            # grasp_mesh = get_gripper_control_points_o3d(grasps[i])
            all_grasp_meshes.extend(grasp_mesh)
            
            # 绘制 grasp_pc 的 7 个点为红色小球
            # points = grasp_pc[i]  # shape: [7, 3]
            # for p in points:
            #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01) # 可以根据需要调整半径
            #     sphere.translate(p)
            #     sphere.paint_uniform_color([1, 0, 0]) # 红色
            #     all_grasp_meshes.append(sphere)
            
            
        # Grasp_confidence_level = np.load("/home/robot/WCL/GraspCoT/preds_list.npy", allow_pickle=True)
        # Grasp_confidence_level = Grasp_confidence_level[10]
        # conf = Grasp_confidence_level.detach().cpu().numpy()
        # top3_idx = np.argsort(conf)[-3:][::-1]
        # for i in range(0, 3):
        #     grasp_mesh = get_gripper_control_points_o3d(grasps[top3_idx[i]])
        #     all_grasp_meshes.extend(grasp_mesh)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd[:, :3])
        point_cloud.colors = o3d.utility.Vector3dVector(pcd[:, 3:]/255.0)
        # point_cloud.paint_uniform_color([0.5, 0.5, 0.5])



        # 创建坐标轴来显示原点
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1,  # 坐标轴长度
            origin=[0, 0, 0]  # 原点位置
        )

        o3d.visualization.draw_geometries([point_cloud] + all_grasp_meshes, 
                                        window_name="Point Cloud with RGB Colors and Origin",
                                        width=1024, 
                                        height=768,
                                        point_show_normal=False)
        
        # show_point_sequential(pc_list, coordinate_frame, vertical_grasp_expanded, save_dir=f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/visual_grasps")
        # show_grasps_sequential(point_cloud, coordinate_frame, grasps, save_dir=f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/visual_grasps")
        # np.save(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/visual_grasps/grasps.npy", grasps)
        


def show_mesh_transparent(mesh: o3d.geometry.TriangleMesh, alpha=0.8):
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    gui.Application.instance.initialize()
    w = gui.Application.instance.create_window("Voxels (20% Transparent)", 1280, 720)

    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(w.renderer)
    w.add_child(scene_widget)

    mat = rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency"
    mat.base_color = (0.7, 0.7, 0.7, float(alpha))  # RGBA, alpha=0.8 -> 20%透明
    mat.base_roughness = 0.9
    mat.base_reflectance = 0.2

    scene_widget.scene.add_geometry("voxels", mesh, mat)

    # 白色背景（尽量兼容）
    try:
        if hasattr(scene_widget.scene, "set_background"):
            scene_widget.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    except Exception:
        pass

    # 太阳光（兼容你之前的版本签名）
    scene = scene_widget.scene.scene
    direction = np.array([[-1.0], [-1.0], [-2.0]], dtype=np.float32)
    color     = np.array([[ 1.0], [ 1.0], [ 1.0]], dtype=np.float32)
    scene.set_sun_light(direction, color, 80000.0)
    scene.enable_sun_light(True)

    bounds = scene_widget.scene.bounding_box
    scene_widget.setup_camera(60.0, bounds, bounds.get_center())

    gui.Application.instance.run()


def draw_voxels_with_gap(pcd: o3d.geometry.PointCloud, voxel_size=0.0125, shrink=0.85, alpha=0.8):
    """
    shrink < 1.0 会产生缝隙；越小缝越大。建议 0.7~0.95
    alpha=0.8 -> 20%透明
    """
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

    s = voxel_size * shrink
    mesh_all = o3d.geometry.TriangleMesh()

    for v in vg.get_voxels():
        center = vg.get_voxel_center_coordinate(v.grid_index)

        box = o3d.geometry.TriangleMesh.create_box(width=s, height=s, depth=s)
        box.translate(center - np.array([s, s, s]) / 2.0)
        box.paint_uniform_color([0.7, 0.7, 0.7])

        mesh_all += box

    mesh_all.compute_vertex_normals()

    # 用新渲染器显示透明度（替代 draw_geometries）
    show_mesh_transparent(mesh_all, alpha=alpha)


def voxel_visualize_with_gap(npy_path, voxel_size=0.0125, shrink=0.85, alpha=0.8):
    pcd_np = np.load(npy_path)
    
    if pcd_np.shape[0] > 4096:
        _, indices = farthest_points(pcd_np[:, :3], 4096, distance_by_translation_point, return_center_indexes=True)
        pcd_np = pcd_np[indices]
        
    xyz = pcd_np[:, :3].astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    draw_voxels_with_gap(pcd, voxel_size=voxel_size, shrink=shrink, alpha=alpha)

if __name__ == "__main__":
    visualize_pc_data()
    # voxel_visualize_with_gap(
    #     "/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/154_spoon/fused_pc_clean.npy",
    #     voxel_size=0.005,
    #     shrink=0.85
    # )