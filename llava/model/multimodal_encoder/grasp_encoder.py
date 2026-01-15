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

# class GraspNet(nn.Module):
#     """
#     Class to encoder the grasping pose.
#     """
#     def __init__(self, d_model=4096, num_heads=8, attn_dropout=0.1):
#         # Initialize parent class with custom parameters directly
#         super(GraspNet, self).__init__()
#         self.initialize(d_model, num_heads, attn_dropout)

#     def initialize(self, d_model, num_heads, attn_dropout):
#         self.type_img   = nn.Parameter(torch.zeros(1, d_model))
#         self.type_voxel = nn.Parameter(torch.zeros(1, d_model))
#         nn.init.normal_(self.type_img,   mean=0.0, std=0.02)
#         nn.init.normal_(self.type_voxel, mean=0.0, std=0.02)

#         self.grasp_mlp = nn.Sequential(
#             nn.Linear(3, 256), nn.GELU(),
#             nn.Linear(256, 1024), nn.GELU(),
#             nn.Linear(1024, d_model)
#         )
        
#         # --- cross-attn: Q attends to memory(K,V) ---
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=d_model, num_heads=num_heads, dropout=attn_dropout, batch_first=True
#         )

#         # optional: stabilize training
#         self.q_norm = nn.LayerNorm(d_model)
#         self.mem_norm = nn.LayerNorm(d_model)

#     def encode_grasps(self, grasps):
#         """
#         grasps: (25, 6, 3)
#         return: (25, D)
#         """
#         x = self.grasp_mlp(grasps)     # (25,6,D)
#         q = x.mean(dim=1)              # (25,D)
#         return q

#     def forward(self, image_features, voxel_features, grasps):
#         grasps = grasps.to(torch.bfloat16)
#         img = image_features + self.type_img          # (Limg,D) + (D,)
#         vox = voxel_features + self.type_voxel        # (Lvox,D)
#         mem = torch.cat([img, vox], dim=0)            # (L, D), L=Limg+Lvox

#         # --- grasp queries ---
#         q = self.encode_grasps(grasps)                # (25,D)

#         # --- cross-attn expects (N, L, D) with batch_first=True ---
#         Q = self.q_norm(q).unsqueeze(0)               # (1,25,D)
#         M = self.mem_norm(mem).unsqueeze(0)           # (1,L,D)

#         out, _ = self.cross_attn(Q, M, M)             # (1,25,D)
#         out = out.squeeze(0)                          # (25,D)
        
#         return out

class GraspNet(nn.Module):
    def __init__(self):
        super(GraspNet, self).__init__()
        self.initialize()

    def initialize(self, d_model=4096, num_heads=8, attn_dropout=0.1):
        self.net = networks.define_classifier(
            1, 0.1, 256, 2, [], "normal", 0.02, "cuda"
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)

    def forward(self, image_features, pc, grasps):
        # print(f"[GraspNet] Initialization check:")
        # for name, param in self.net.named_parameters():
        #     print(f"[GraspNet] Param: {name}, Mean: {param.data.abs().mean().item():.6f}, Std: {param.data.std().item():.6f}")

        with torch.cuda.amp.autocast(enabled=False):
            result = self.net.encode(pc, grasps)
        
            # Ensure result has the same dtype as image_features for attention
            image_features = image_features.to(dtype=result.dtype)

            query = result.unsqueeze(0)
            key_value = image_features.unsqueeze(0)

            query = self.q_norm(query)
            key_value = self.kv_norm(key_value)

            # Cross attention: query attends to key_value
            attn_output, _ = self.cross_attn(query, key_value, key_value)

            return attn_output.squeeze(0)

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