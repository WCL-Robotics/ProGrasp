import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from torch.autograd import Function
from torch.nn.modules.utils import _pair

from mmdet3d.registry import MODELS
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape, dynamic_scatter_3d
# from .unproject import voxelize as voxelize_for_scatter
from position_contrained_6dof_graspnet.models import networks
import os

# class GraspNet(networks.GraspSamplerVAE):
#     """
#     Class to encoder the grasping pose.
#     """
#     def __init__(self):
#         # Initialize parent class with custom parameters directly
#         super(GraspNet, self).__init__(
#             model_scale=1, 
#             pointnet_radius=0.3, 
#             pointnet_nclusters=128, 
#             latent_size=2, 
#             device="cuda"
#         )
#         # Initialize weights for the current instance (self)
#         networks.init_weights(self, "normal", 0.02)

#     def encode(self, pc_xyz, grasps):
#         with torch.cuda.amp.autocast(enabled=False):
#             pc_xyz_expand = pc_xyz.unsqueeze(0).repeat(grasps.shape[0], 1, 1)
#             pc_xyz_expand = pc_xyz_expand.contiguous().to(dtype=torch.float32)
#             grasp_features = grasps.unsqueeze(1).expand(-1, pc_xyz_expand.shape[1], -1)
#             features = torch.cat(
#                 (pc_xyz_expand, grasp_features),
#                 -1)

#             # features = torch.cat((features, position_contraint_feature), -1)
#             features = features.transpose(-1, 1).contiguous()
            
#             for module in self.encoder[0]:
#                 pc_xyz_expand, features = module(pc_xyz_expand, features)
#                 print(f"[DEBUG] After Layer: Features Min={features.min().item():.8f}, Max={features.max().item():.8f}, Mean={features.mean().item():.4f}")
        
#         temp = self.encoder[1](features.squeeze(-1))
#         return self.encoder[1](features.squeeze(-1))

#     def forward(self, pc, grasps):
#         return self.encode(pc, grasps)

class GraspNet(nn.Module):
    def __init__(self):
        super(GraspNet, self).__init__()
        self.initialize()

    def initialize(self):
        self.net = networks.define_classifier(
            1, 0.1, 512, 2, [], "normal", 0.02, "cuda"
        )

    def forward(self, pc, grasps):
        # print(f"[GraspNet] Initialization check:")
        # for name, param in self.net.named_parameters():
        #     print(f"[GraspNet] Param: {name}, Mean: {param.data.abs().mean().item():.6f}, Std: {param.data.std().item():.6f}")

        with torch.cuda.amp.autocast(enabled=False):
            result = self.net.encode(pc, grasps)
        return result

if __name__ == "__main__":
    # task_obj = '/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans'
    # folders = [f for f in os.listdir(task_obj) if os.path.isdir(os.path.join(task_obj, f))]
    # folders_sorted = sorted(folders, key=lambda s: int(s[:3]))
    # pc_ = np.load("/home/robot/WCL/GraspCoT/pc.npy")
    # for file in folders_sorted:
    #     pcd = np.load(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/down_pc_4096.npy")
    #     pcd = pcd.astype(np.float32)
    #     if np.allclose(pc_, pcd, atol=1e-5):
    #         print(f"Found matching point cloud in folder: {file}")
            
    net = networks.define_classifier(1, 0.02, 1024, 2, [], "normal", 0.02, "cuda")
    net.cuda()
    pc = np.load("/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/044_mixing_bowl/down_pc_4096.npy")
    grasps = np.load("/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/044_mixing_bowl/visual_grasps/grasps.npy")
    grasps = grasps.reshape(grasps.shape[0], -1)
    pc = torch.from_numpy(pc).float().cuda()
    grasps = torch.from_numpy(grasps).float().cuda()
    net.encode(pc, grasps)