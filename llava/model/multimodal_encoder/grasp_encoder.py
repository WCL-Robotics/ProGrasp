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
#         self.d_model = d_model
#         # Use an intermediate dimension for the internal Grasp Encoder
#         # This prevents projecting 3D coordinates directly to 4096D, which can be unstable
#         self.inner_dim = 512 
        
#         self.type_img   = nn.Parameter(torch.zeros(1, d_model))
#         self.type_voxel = nn.Parameter(torch.zeros(1, d_model))
#         nn.init.normal_(self.type_img,   mean=0.0, std=0.02)
#         nn.init.normal_(self.type_voxel, mean=0.0, std=0.02)

        
#         # --- cross-attn: Q attends to memory(K,V) ---
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=d_model, num_heads=num_heads, dropout=attn_dropout, batch_first=True
#         )

#         # optional: stabilize training
#         self.q_norm = nn.LayerNorm(d_model)
#         self.mem_norm = nn.LayerNorm(d_model)

#         # ---- small Transformer encoder for 6 points (runs in inner_dim) ----
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=self.inner_dim,
#             nhead=8,
#             dim_feedforward=4 * self.inner_dim,
#             dropout=0,
#             batch_first=True,
#             activation="gelu"
#         )
#         self.grasp_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

#         self.shared_pos_encoder = nn.Sequential(
#             nn.Linear(3, 256),       
#             nn.GELU(),            
#             nn.Linear(256, 1024),    
#             nn.GELU(),
#             nn.Linear(1024, d_model), 
#             nn.LayerNorm(d_model)
#         )

#         self.shape_mlp = nn.Sequential(
#             nn.Linear(3, 128), nn.GELU(),
#             nn.Linear(128, self.inner_dim), 
#             nn.LayerNorm(self.inner_dim)
#         )
#         self.shape_up_proj = nn.Sequential(
#             nn.Linear(self.inner_dim, d_model), nn.GELU(),
#             nn.Linear(d_model, d_model),
#             nn.LayerNorm(d_model)
#         )

#         # ---- learnable positional embedding for keypoints (inner_dim) ----
#         self.grasp_pos_embed = nn.Parameter(torch.zeros(1, 6, self.inner_dim))
#         nn.init.normal_(self.grasp_pos_embed, mean=0.0, std=0.02)

#     def encode_grasps(self, grasps):
#         """
#         grasps: (B, 6, 3)
#         return: (B, D_model)
#         """
#         grasps = grasps.float()
        
#         # 1. 分离中心点和相对形状
#         center = grasps.mean(dim=1, keepdim=True)   # (B, 1, 3)
#         grasps_rel = grasps - center                # (B, 6, 3) relative to center

#         # =========================================================
#         # A. 形状流 (Shape Stream) - 在低维 inner_dim 处理
#         # =========================================================
#         x = self.shape_mlp(grasps_rel)              # (B, 6, inner_dim)
#         x = x + self.grasp_pos_embed                # add learnable keypoint embeddings
#         x = self.grasp_encoder(x)                   # (B, 6, inner_dim)
        
#         # 聚合特征: Max pooling 捕捉最显著的几何特征
#         shape_feat_inner = torch.max(x, dim=1)[0]   # (B, inner_dim)
        
#         # 升维到 d_model
#         shape_feat = self.shape_up_proj(shape_feat_inner) # (B, d_model)

#         # =========================================================
#         # B. 位置流 (Position Stream) - 使用共享编码器
#         # =========================================================
#         # 这里直接复用 self.shared_pos_encoder
#         center_squeeze = center.squeeze(1)          # (B, 3)
#         pos_feat = self.shared_pos_encoder(center_squeeze) # (B, d_model)

#         # =========================================================
#         # C. 融合 (Fusion)
#         # =========================================================
#         # 类似于 Transformer 的 Embedding + PosEmbedding
#         q = shape_feat + pos_feat                   # (B, d_model)
        
#         return q


#     def forward(self, image_features, voxel_features, voxel_centers, grasps):
#         # grasps = grasps.to(torch.bfloat16)
#         img = image_features + self.type_img          # (Limg,D) + (D,)
        
#         # Add 3D Positional Encoding to Voxel Features
#         if voxel_centers is not None:
#              # Encode centers to inner_dim then project up to d_model using DEDICATED projector
#              # This shares the same coordinate embedding space (pos_mlp) as grasp centers
#              vox_pos = self.shared_pos_encoder(voxel_centers.float())
#              vox = voxel_features + self.type_voxel + vox_pos
#         else:
#              vox = voxel_features + self.type_voxel
             
#         mem = torch.cat([img, vox], dim=0)            # (L, D)

#         # --- grasp queries (already projected to d_model) ---
#         q = self.encode_grasps(grasps)                # (25, d_model)

#         # --- cross-attn (d_model space) ---
#         Q = self.q_norm(q).unsqueeze(0)               # (1,25,D)
#         M = self.mem_norm(mem).unsqueeze(0)           # (1,L,D)

#         out, _ = self.cross_attn(Q, M, M)             # (1,25,D)
#         out = out.squeeze(0)                          # (25,D)
        
#         return out


# class GraspNet(nn.Module):
#     """
#     Class to encoder the grasping pose.
#     """
#     def __init__(self, d_model=4096, num_heads=8, attn_dropout=0.1):
#         # Initialize parent class with custom parameters directly
#         super(GraspNet, self).__init__()
#         self.initialize(d_model, num_heads, attn_dropout)

#     def initialize(self, d_model, num_heads, attn_dropout):
#         self.d_model = d_model
#         # Use an intermediate dimension for the internal Grasp Encoder
#         # This prevents projecting 3D coordinates directly to 4096D, which can be unstable
#         self.inner_dim = 512 
        
#         self.type_img   = nn.Parameter(torch.zeros(1, d_model))
#         self.type_voxel = nn.Parameter(torch.zeros(1, d_model))
#         nn.init.normal_(self.type_img,   mean=0.0, std=0.02)
#         nn.init.normal_(self.type_voxel, mean=0.0, std=0.02)

#         # MLP for encoding geometric shape (Relative Coordinates) -> inner_dim
#         self.shape_mlp = nn.Sequential(
#             nn.Linear(3, 128), nn.GELU(),
#             nn.Linear(128, self.inner_dim), nn.LayerNorm(self.inner_dim)
#         )

#         # MLP for encoding 3D positions (Grasp Center & Voxel Centers) -> inner_dim
#         self.pos_mlp = nn.Sequential(
#             nn.Linear(3, 128), nn.GELU(),
#             nn.Linear(128, self.inner_dim), nn.LayerNorm(self.inner_dim)
#         )

#         # 输入是 Shape(inner_dim) + Position(inner_dim)
#         self.grasp_up_proj = nn.Sequential(
#             nn.Linear(self.inner_dim * 2, self.inner_dim * 2), nn.GELU(),
#             nn.Linear(self.inner_dim * 2, d_model),
#             nn.LayerNorm(d_model)
#         )
#         # Projector for Voxel Position features (Key/Value Position)
#         self.pos_up_proj = nn.Sequential(
#             nn.Linear(self.inner_dim, d_model),
#             nn.LayerNorm(d_model),
#         )
        
#         # --- cross-attn: Q attends to memory(K,V) ---
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=d_model, num_heads=num_heads, dropout=attn_dropout, batch_first=True
#         )

#         # optional: stabilize training
#         self.q_norm = nn.LayerNorm(d_model)
#         self.mem_norm = nn.LayerNorm(d_model)

#         # ---- small Transformer encoder for 6 points (runs in inner_dim) ----
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=self.inner_dim,
#             nhead=8,
#             dim_feedforward=4 * self.inner_dim,
#             dropout=0,
#             batch_first=True,
#             activation="gelu"
#         )
#         self.grasp_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

#         # ---- learnable positional embedding for keypoints (inner_dim) ----
#         self.grasp_pos_embed = nn.Parameter(torch.zeros(1, 6, self.inner_dim))
#         nn.init.normal_(self.grasp_pos_embed, mean=0.0, std=0.02)


#     # def encode_grasps(self, grasps):
#     #     """
#     #     grasps: (25, 6, 3)
#     #     return: (25, D)
#     #     """
#     #     x = self.grasp_mlp(grasps)     # (25,6,D)
#     #     q = x.mean(dim=1)              # (25,D)
#     #     return q

#     def encode_grasps(self, grasps):
#         """
#         grasps: (B, 6, 3)
#         return: (B, D_model)
#         """
#         grasps = grasps.float()
        
#         # 1. Normalize Coordinates
#         center = grasps.mean(dim=1, keepdim=True)   # (B, 1, 3)
#         grasps_rel = grasps - center                # (B, 6, 3)

#         # 2. Encode Shape -> inner_dim
#         x = self.shape_mlp(grasps_rel)              # (B, 6, inner_dim)
#         x = x + self.grasp_pos_embed                # (B, 6, inner_dim)
#         x = self.grasp_encoder(x)                   # (B, 6, inner_dim)

#         shape_feat = torch.max(x, dim=1)[0]         # (B, inner_dim)

#         pos_feat = self.pos_mlp(center).squeeze(1)  # (B, inner_dim)

#         global_feat = torch.cat([shape_feat, pos_feat], dim=-1) # (B, 2*inner_dim)
#         q = self.grasp_up_proj(global_feat)         # (B, d_model)
#         return q

#     def forward(self, image_features, voxel_features, voxel_centers, grasps):
#         # grasps = grasps.to(torch.bfloat16)
#         img = image_features + self.type_img          # (Limg,D) + (D,)
        
#         # Add 3D Positional Encoding to Voxel Features
#         if voxel_centers is not None:
#              # Encode centers to inner_dim then project up to d_model using DEDICATED projector
#              # This shares the same coordinate embedding space (pos_mlp) as grasp centers
#              vox_pos_inner = self.pos_mlp(voxel_centers.float())  # (Lvox, inner_dim)
#              vox_pos = self.pos_up_proj(vox_pos_inner)            # (Lvox, d_model)
#              vox = voxel_features + self.type_voxel + vox_pos
#         else:
#              vox = voxel_features + self.type_voxel
             
#         mem = torch.cat([img, vox], dim=0)            # (L, D)

#         # --- grasp queries (already projected to d_model) ---
#         q = self.encode_grasps(grasps)                # (25, d_model)

#         # --- cross-attn (d_model space) ---
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
            1, 0.1, 512, 2, [], "normal", 0.02, "cuda"
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
            # image_features is a list of variable length tensors, we need to pad them
            image_features = [f.to(dtype=result.dtype) for f in image_features]
            
            lengths = [f.shape[0] for f in image_features]
            max_len = max(lengths)
            batch_size = len(image_features)

            # Create key_padding_mask: True indicates the position is padding and should be ignored
            key_padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=result.device)
            for i, length in enumerate(lengths):
                key_padding_mask[i, :length] = False

            # Pad the sequence to (B, max_len, D)
            image_features_padded = torch.nn.utils.rnn.pad_sequence(image_features, batch_first=True)

            # result: (B, D) -> query: (B, 1, D)
            query = result.unsqueeze(1)
            key_value = image_features_padded

            query = self.q_norm(query)
            key_value = self.kv_norm(key_value)

            # Cross attention: query attends to key_value
            attn_output, _ = self.cross_attn(query, key_value, key_value, key_padding_mask=key_padding_mask)

            out = result.unsqueeze(1) + attn_output
            return out

    # def forward(self, image_features, pc, grasps):
    #     # print(f"[GraspNet] Initialization check:")
    #     # for name, param in self.net.named_parameters():
    #     #     print(f"[GraspNet] Param: {name}, Mean: {param.data.abs().mean().item():.6f}, Std: {param.data.std().item():.6f}")

    #     with torch.cuda.amp.autocast(enabled=False):
    #         result = self.net.encode(pc, grasps)
        
    #         # Ensure result has the same dtype as image_features for attention
    #         image_features = image_features.to(dtype=result.dtype)

    #         query = result.unsqueeze(0)
    #         key_value = image_features.unsqueeze(0)

    #         query = self.q_norm(query)
    #         key_value = self.kv_norm(key_value)

    #         # Cross attention: query attends to key_value
    #         attn_output, _ = self.cross_attn(query, key_value, key_value)

    #         return attn_output.squeeze(0)

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