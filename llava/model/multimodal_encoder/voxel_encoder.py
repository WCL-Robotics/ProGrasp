import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.autograd import Function
from torch.nn.modules.utils import _pair

from mmdet3d.registry import MODELS
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape, dynamic_scatter_3d
# from .unproject import voxelize as voxelize_for_scatter
import inspect

class PillarVoxelTower(nn.Module):
    def __init__(self, args):
        super().__init__()
        point_cloud_range = [-0.5, -0.5, 0, 0.5, 0.5, 1]
        # change for real
        self.voxel_type = 'hard'
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=[0.025, 0.025, 1.0],
            max_voxels=(16000, 40000))

        voxel_encoder=dict(
            type='PillarFeatureNet',
            in_channels=6,
            feat_channels=[256],
            with_distance=False,
            voxel_size=[0.025, 0.025, 1.0],
            point_cloud_range=point_cloud_range)

        middle_encoder=dict(
            type='PointPillarsScatter', in_channels=256, output_shape=[40, 40])

        self.voxel_layer = VoxelizationByGridShape(**voxel_layer)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)     
        self.conv_proj = nn.Sequential(
            nn.Conv2d(256, 1024, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 4096, 1, stride=2, padding=0))


    @torch.no_grad()
    def forward(self, points):

        points_new = [pc.to(torch.float32) for pc in points]
        voxel_dict = self.voxelize(points_new)
        voxel_features = self.voxel_encoder(voxel_dict['voxels'].to(torch.bfloat16),
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1

        spatial_features = self.middle_encoder(voxel_features, voxel_dict['coors'],
                            batch_size)


        spatial_features = self.conv_proj(spatial_features)
        B, C, H, W = spatial_features.shape
        spatial_features = spatial_features.view(B, C, H* W)
        new_spatial_features, batch_offset = [], []
        for i in range(B):
            valid_inds = (spatial_features[i].sum(0) != 0)
            valid_spatial_features = spatial_features[i][:, valid_inds]
            new_spatial_features.append(valid_spatial_features.permute(1,0).to(torch.bfloat16))
            batch_offset.append(valid_spatial_features.shape[-1]+1)

        return new_spatial_features, batch_offset



    @torch.no_grad()
    def voxelize(self, points):
        """Apply voxelization to point cloud.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (list[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            Dict[str, Tensor]: Voxelization information.

            - voxels (Tensor): Features of voxels, shape is MxNxC for hard
              voxelization, NxC for dynamic voxelization.
            - coors (Tensor): Coordinates of voxels, shape is Nx(1+NDim),
              where 1 represents the batch index.
            - num_points (Tensor, optional): Number of points in each voxel.
            - voxel_centers (Tensor, optional): Centers of voxels.
        """

        voxel_dict = dict()

        if self.voxel_type == 'hard':
            voxels, coors, num_points, voxel_centers = [], [], [], []
            for i, res in enumerate(points):
                res_voxels, res_coors, res_num_points = self.voxel_layer(res)
                res_voxel_centers = (
                    res_coors[:, [2, 1, 0]] + 0.5) * res_voxels.new_tensor(
                        self.voxel_layer.voxel_size) + res_voxels.new_tensor(
                            self.voxel_layer.point_cloud_range[0:3])
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                voxels.append(res_voxels)
                coors.append(res_coors)
                num_points.append(res_num_points)
                voxel_centers.append(res_voxel_centers)

            voxels = torch.cat(voxels, dim=0)
            coors = torch.cat(coors, dim=0)
            num_points = torch.cat(num_points, dim=0)
            voxel_centers = torch.cat(voxel_centers, dim=0)

            voxel_dict['num_points'] = num_points
            voxel_dict['voxel_centers'] = voxel_centers
        elif self.voxel_type == 'dynamic':
            coors = []
            # dynamic voxelization only provide a coors mapping
            for i, res in enumerate(points):
                res_coors = self.voxel_layer(res)
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                coors.append(res_coors)
            voxels = torch.cat(points, dim=0)
            coors = torch.cat(coors, dim=0)

        else:
            raise ValueError(f'Invalid voxelization type {self.voxel_type}')

        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors

        return voxel_dict

    @property
    def dtype(self):
        return self.dtype

    @property
    def device(self):
        return self.device



class SecondVoxelTower(nn.Module):
    def __init__(self, args):
        super().__init__()
        # point_cloud_range = [-0.5, -0.5, 0, 0.5, 0.5, 1]
        point_cloud_range = [-0.25, -0.25, 0, 0.25, 0.25, 0.5]
        self.voxel_type = 'hard'
        voxel_layer=dict(
            max_num_points=64,
            point_cloud_range=point_cloud_range,
            voxel_size = [0.00625, 0.00625, 0.00625],
            max_voxels=(16000, 40000))

        voxel_encoder=dict(type='HardSimpleVFE')
        
        middle_encoder=dict(
            type='SparseEncoder',
            in_channels=3,
            output_channels=4096,
            # output_channels=2048,
            sparse_shape=[80, 80, 80],
            order=('conv', 'norm', 'act'))

        self.voxel_layer = VoxelizationByGridShape(**voxel_layer)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)
        print("source file:", inspect.getsourcefile(type(self.middle_encoder)))

        # print(f"\n[SecondVoxelTower] Initialized parameters:")
        # for name, param in self.named_parameters():
        #     print(f"  {name}: {param.shape}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")
     

    def stats_pointcloud(self, points_list, q_low=0.01, q_high=0.99):
        for b, pc in enumerate(points_list):
            pc = pc.to(torch.float32)
            assert pc.dim() == 2 and pc.size(1) >= 3, f"pc shape wrong: {pc.shape}"
            xyz = pc[:, :3]

            # 去掉 NaN/Inf（如果你确定没有，可以删掉）
            mask = torch.isfinite(xyz).all(dim=1)
            xyz = xyz[mask]
            if xyz.numel() == 0:
                print(f"[B{b}] empty after finite filter")
                continue

            mn = xyz.min(dim=0).values
            mx = xyz.max(dim=0).values
            size = mx - mn

            q1 = torch.quantile(xyz, q_low, dim=0)
            q2 = torch.quantile(xyz, q_high, dim=0)
            qsize = q2 - q1

            print(
                f"[B{b}] N={xyz.shape[0]} | "
                f"min={mn.tolist()} max={mx.tolist()} size={size.tolist()} | "
                f"q{int(q_low*100)}={q1.tolist()} q{int(q_high*100)}={q2.tolist()} qsize={qsize.tolist()}"
            )

    # @torch.no_grad()
    def forward(self, points):
        with torch.cuda.amp.autocast(enabled=False):
            
            # voxel_dict = self.voxelize(points)
            # Convert points to float32 for voxelization (mmdet3d doesn't support bfloat16)
            points_float32 = [pc.to(torch.float32) for pc in points]

            voxel_dict = self.voxelize(points_float32)



            voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                                voxel_dict['num_points'],
                                                voxel_dict['coors'])
            batch_size = voxel_dict['coors'][-1, 0].item() + 1

            spatial_features = self.middle_encoder(voxel_features.to(torch.float32), voxel_dict['coors'].to(torch.int32),
                                batch_size)

            
            B, C, D, H, W = spatial_features.shape
            spatial_features = spatial_features.view(B, C, D * H* W)
            # B, C, H, W = spatial_features.shape
            # spatial_features = spatial_features.view(B, C, H* W)
            new_spatial_features, batch_offset = [], []
            for i in range(B):
                valid_inds = spatial_features[i].abs().sum(0) > 0
                # print(f"[B{i}] valid tokens:", int(valid_inds.sum().item()))
                valid_spatial_features = spatial_features[i][:, valid_inds]
                new_spatial_features.append(valid_spatial_features.permute(1,0).to(torch.bfloat16))
                batch_offset.append(valid_spatial_features.shape[-1]+1)

            return new_spatial_features, batch_offset



    # @torch.no_grad()
    def voxelize(self, points):
        """Apply voxelization to point cloud.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (list[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            Dict[str, Tensor]: Voxelization information.

            - voxels (Tensor): Features of voxels, shape is MxNxC for hard
              voxelization, NxC for dynamic voxelization.
            - coors (Tensor): Coordinates of voxels, shape is Nx(1+NDim),
              where 1 represents the batch index.
            - num_points (Tensor, optional): Number of points in each voxel.
            - voxel_centers (Tensor, optional): Centers of voxels.
        """

        voxel_dict = dict()

        if self.voxel_type == 'hard':
            voxels, coors, num_points, voxel_centers = [], [], [], []
            for i, res in enumerate(points):
                res_voxels, res_coors, res_num_points = self.voxel_layer(res)
                res_voxel_centers = (
                    res_coors[:, [2, 1, 0]] + 0.5) * res_voxels.new_tensor(
                        self.voxel_layer.voxel_size) + res_voxels.new_tensor(
                            self.voxel_layer.point_cloud_range[0:3])
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                voxels.append(res_voxels)
                coors.append(res_coors)
                num_points.append(res_num_points)
                voxel_centers.append(res_voxel_centers)

            voxels = torch.cat(voxels, dim=0)
            coors = torch.cat(coors, dim=0)
            num_points = torch.cat(num_points, dim=0)
            voxel_centers = torch.cat(voxel_centers, dim=0)

            voxel_dict['num_points'] = num_points
            voxel_dict['voxel_centers'] = voxel_centers
        elif self.voxel_type == 'dynamic':
            coors = []
            # dynamic voxelization only provide a coors mapping
            for i, res in enumerate(points):
                res_coors = self.voxel_layer(res)
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                coors.append(res_coors)
            voxels = torch.cat(points, dim=0)
            coors = torch.cat(coors, dim=0)

        else:
            raise ValueError(f'Invalid voxelization type {self.voxel_type}')

        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors

        return voxel_dict

    # @property
    # def dummy_feature(self):
    #     return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    # @property
    # def dtype(self):
    #     return self.dtype

    # @property
    # def device(self):
    #     return self.device


def dataset_xyz_quantile(
    scans_root: str,
    q_low: float = 0.05,
    q_high: float = 0.95,
    sample_per_cloud: int = 0,   # 0 表示不采样：用全部点；>0 表示每个点云最多随机采样这么多点
    max_total_points: int = 0    # 0 表示不限制；>0 表示全局最多保留这么多点（到达后停止继续收集）
):
    # 1) 收集所有 down_pc_4096.npy 路径
    folders = [f for f in os.listdir(scans_root) if os.path.isdir(os.path.join(scans_root, f))]
    folders_sorted = sorted(folders, key=lambda s: int(s[:3]) if s[:3].isdigit() else s)

    all_xyz = []
    global_min = torch.full((3,), float("inf"))
    global_max = torch.full((3,), float("-inf"))
    total_kept = 0
    total_seen = 0

    for folder in tqdm(folders_sorted, desc="Scanning point clouds"):
        pth = os.path.join(scans_root, folder, "down_pc_4096.npy")
        if not os.path.exists(pth):
            continue

        pcd = np.load(pth)              # (4096, 3) 或 (4096, >=3)
        if pcd.ndim != 2 or pcd.shape[1] < 3:
            continue

        xyz = torch.from_numpy(pcd[:, :3]).to(torch.float32)
        mask = torch.isfinite(xyz).all(dim=1)
        xyz = xyz[mask]

        if xyz.numel() == 0:
            continue

        total_seen += xyz.shape[0]

        # 更新全局 min/max（不做分位过滤）
        global_min = torch.minimum(global_min, xyz.min(dim=0).values)
        global_max = torch.maximum(global_max, xyz.max(dim=0).values)

        # 2) 可选：每个点云内采样
        if sample_per_cloud and xyz.shape[0] > sample_per_cloud:
            idx = torch.randperm(xyz.shape[0])[:sample_per_cloud]
            xyz = xyz[idx]

        # 3) 可选：全局限制最大点数
        if max_total_points and (total_kept + xyz.shape[0]) > max_total_points:
            remain = max_total_points - total_kept
            if remain <= 0:
                break
            idx = torch.randperm(xyz.shape[0])[:remain]
            xyz = xyz[idx]

        all_xyz.append(xyz)
        total_kept += xyz.shape[0]

        if max_total_points and total_kept >= max_total_points:
            break

    if len(all_xyz) == 0:
        raise RuntimeError("No valid point clouds found!")

    xyz_all = torch.cat(all_xyz, dim=0)  # (N, 3)

    # 4) 计算全局分位数范围（覆盖中间 90%）
    q1 = torch.quantile(xyz_all, q_low, dim=0)
    q2 = torch.quantile(xyz_all, q_high, dim=0)

    print("\n===== Dataset XYZ Stats =====")
    print(f"Total seen points (finite): {total_seen}")
    print(f"Total kept points for quantile: {xyz_all.shape[0]}")
    print(f"Global min: {global_min.tolist()}")
    print(f"Global max: {global_max.tolist()}")
    print(f"q{int(q_low*100)}  (min for central 90%): {q1.tolist()}")
    print(f"q{int(q_high*100)} (max for central 90%): {q2.tolist()}")
    print(f"Central-90% size (q_high-q_low): {(q2-q1).tolist()}")

    return {
        "global_min": global_min,
        "global_max": global_max,
        "q_low": q1,
        "q_high": q2,
        "n_seen": total_seen,
        "n_kept": xyz_all.shape[0],
    }


if __name__ == "__main__":
    import os
    import numpy as np
    import torch
    from tqdm import tqdm
    task_obj = "/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans"

    # 方案1：用全部点（如果数据量不大）
    # stats = dataset_xyz_quantile(task_obj, q_low=0.05, q_high=0.95, sample_per_cloud=0, max_total_points=0)

    # 方案2：更稳的做法：每个点云采样 1024 点，总点数最多 2,000,000（推荐，省内存且分位足够准）
    stats = dataset_xyz_quantile(
        task_obj,
        q_low=0.05, q_high=0.95,
        sample_per_cloud=1024,
        max_total_points=2_000_000
    )