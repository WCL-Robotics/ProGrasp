import torch
import torch.nn as nn

from .video_processor import RGBDVideoProcessor
from .spatial_aware_module import SpatialAwareModule
from .unproject import backprojector_dataloader, voxelize, voxelize_points, interpolate_feat_up, interpolate_xyz_down
from pytorch3d.ops import sample_farthest_points
from torch_scatter import scatter_mean, scatter_sum
from .position_encodings import PositionEmbeddingLearnedMLP
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt


class PromptEncoder(nn.Module):
    
    def __init__(self, latent_dim=4096):
        super(PromptEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.pos_emb3d = PositionEmbeddingLearnedMLP(dim=3, num_pos_feats=latent_dim)

    def encode_pe(self, xyz=None):
        return self.pos_emb3d(xyz)
    
    def forward(self, clicks):
        # (n, 3)
        pos_embed = self.encode_pe(clicks) #  (N, F)
        return pos_embed

class RGBDVideoTower(nn.Module):
    def __init__(self, vision_tower, video_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.num_frames = args.num_frames
        self.num_sample_tokens = args.num_sample_tokens
        self.pooling = 'voxelize'
        # self.voxel_size = 0.2
        # self.voxel_size = 0.025
        self.voxel_size = 0.0125
        # self.voxel_size = 0.00625
        self.vision_tower_name = vision_tower
        self.video_tower_name = video_tower

        # self.pc_range = [-0.5, -0.5, 0, 0.5, 0.5, 1]
        self.pc_range = [-0.25, -0.25, 0, 0.25, 0.25, 0.5]

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_video_tower', False):
            self.load_model()
        else:
            self.cfg_only = None
        
        self.initialize()

        # print(f"\n[RGBDVideoTower] Initialized parameters:")
        # for name, param in self.named_parameters():
        #     print(f"  {name}: {param.shape}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.video_tower_name))
            return

        # self.video_processor = RGBDVideoProcessor(self.vision_tower_name, self.num_frames)
        if self.video_tower_name == 'SpatialAwareModule':
            self.video_tower = SpatialAwareModule()
        else:
            raise NotImplementedError

        self.prompt_encoder = PromptEncoder()
        # self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def initialize(self):
        # Initialize Self-Attention for Voxels
        # d_model=1024 matches the feature dimension from Vision Tower (e.g., CLIP-Large)
        self.voxel_self_attn = nn.TransformerEncoderLayer(
            d_model=1024, 
            nhead=8, 
            dim_feedforward=2048, 
            dropout=0.1, 
            activation='gelu', 
            batch_first=True
        )
        self.geo_embed_layer = nn.Sequential(
            nn.Linear(3+1, 256),       
            nn.GELU(),            
            nn.Linear(256, 512),    
            nn.GELU(),
            nn.Linear(512, 1024), 
            nn.LayerNorm(1024)
        )
        self.voxel_self_attn = self.voxel_self_attn.to(dtype=torch.bfloat16)
        self.video_tower.initialize()

    def forward(self, features, depths, poses, intrinsics, lengths=None, grasps=None):
        """
        Compute visual features/position embeddings for each patch.

        Args:
            - features: (B, V, 1024, 24, 24), image token features
            - depths: (B, V, 336, 336), depth images
            - poses: (B, V, 4, 4) pose information
            - instrinsics: (B, V, 4, 4), intriniscs
            - lengths: (B,)  view number of each scene

        Returns:
            - rgb_feats_pyramid: [(B, ncam, F, H_i, W_i)]
            - pcd_pyramid: [(B, ncam * H_i * W_i, 3)]
        """
        B, V, C, H, W = features.shape
        assert intrinsics.dim() == 4

        feat_xyz, xyz = backprojector_dataloader([features.flatten(0, 1)], depths, poses, intrinsics)
        # print("[Visualizing xyz in VideoEncoder]")
        # pcd = o3d.geometry.PointCloud()
        # # xyz shape likely (B, N, 3) or (B*V, N, 3). Flatten to (M, 3) for valid parts
        # xyz = xyz[0][0].to(torch.float32)
        # xyz = xyz.reshape(336*336,3)
        # pcd_np = xyz.detach().cpu().numpy().reshape(-1, 3)
        # # 简单过滤掉全0点或无效点以便看得更清楚
        # valid_mask = np.abs(pcd_np).sum(axis=1) > 1e-6
        # pcd.points = o3d.utility.Vector3dVector(pcd_np[valid_mask])
        
        # # 添加坐标轴以便观察方位
        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd, coordinate_frame], window_name="XYZ Debug Visualization")

        valid_mask_feat = ~torch.isnan(feat_xyz[..., 0:1]) # (B, V, H, W, 1)
        feat_xyz_clean = torch.where(valid_mask_feat, feat_xyz, torch.tensor(0.0, device=feat_xyz.device, dtype=feat_xyz.dtype))
        video_features = self.video_tower([features.flatten(0, 1)], [feat_xyz_clean.flatten(0, 1)], (B, V), valid_mask_feat.flatten(0, 1))[0]

        mask24 = valid_mask_feat.squeeze(-1).to(torch.float32)            # (B,V,24,24)
        # 插值到 56×56（用 nearest）
        mask56 = torch.nn.functional.interpolate(
            mask24.flatten(0, 1).unsqueeze(1),  # (B*V,1,24,24)
            size=(56, 56),
            mode="nearest"
        ).squeeze(1).reshape(B, V, 56, 56)     # (B,V,56,56)
        valid_mask_down = mask56.to(torch.bool)  # (B,V,56,56)

        # (B, V*H*W, C)
        # video_features = self.video_tower([features.flatten(0, 1)], [feat_xyz.flatten(0, 1)], (B, V))[0]
        video_features = video_features.reshape(B*V, H, W, C).contiguous()
        video_features_up = interpolate_feat_up(video_features.clone()).reshape(B, -1, C)
        xyz_down = interpolate_xyz_down(xyz.clone())


        valid_inds = (xyz_down[..., 0] > self.pc_range[0]) & (xyz_down[..., 0] < self.pc_range[3]) & \
            (xyz_down[..., 1] > self.pc_range[1]) & (xyz_down[..., 1] < self.pc_range[4]) & \
                (xyz_down[..., 2] > self.pc_range[2]) & (xyz_down[..., 2] < self.pc_range[5])
        valid_inds = valid_inds & valid_mask_down
        # xyz_down[~valid_inds] = xyz_down[~valid_inds] * 0
        # xyz_down[~valid_inds] = 2.0
        # video_features_up[~valid_inds.view(B, -1)] = video_features_up[~valid_inds.view(B, -1)] * 0

        video_xyz = feat_xyz.reshape(B, V*H*W, 3)
        if lengths is not None:
            lengths = lengths*H*W
        if self.pooling == 'fps':
            grasp_features = None
            if self.num_sample_tokens < video_features.shape[1]:
                _, indexs = sample_farthest_points(video_xyz, lengths=lengths, K=self.num_sample_tokens)
                pooled_video_features = torch.gather(video_features, 1, indexs.unsqueeze(2).expand(B, self.num_sample_tokens, C))
            else:
                pooled_video_features = video_features
            batch_offset = None
        elif self.pooling == 'voxelize':
            # p2v = voxelize(xyz_down, self.voxel_size)  # （B, N)
            # pooled_video_features = torch.cat([scatter_mean(video_features_up[b], p2v[b], dim=0) for b in range(len(video_features_up))]) # bn, F
            # batch_offset = ((p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
            xyz_flat  = xyz_down.view(B, -1, 3)
            feat_flat = video_features_up.view(B, -1, C)
            valid_flat = valid_inds.view(B, -1)
            batch_features_list = []
            batch_counts = []
            
            for b in range(len(video_features_up)):
                m = valid_flat[b]  # (N,)
                xyz_b  = xyz_flat[b][m]      # (Nv, 3)

                feat_b = feat_flat[b][m]     # (Nv, C)
                # 只对有效点 voxelize
                p2v_b = voxelize_points(xyz_b.unsqueeze(0), self.voxel_size)[0]  # (Nv,)
                v_xyz_b = scatter_mean(xyz_b, p2v_b, dim=0)
                
                # 计算点数密度 (Density)
                # 创建一个全1的tensor，scatter_sum 后就是每个体素里的点数
                ones_b = torch.ones(p2v_b.shape[0], 1, device=p2v_b.device, dtype=feat_b.dtype)
                v_count_b = scatter_sum(ones_b, p2v_b, dim=0)  # (num_vox, 1)
                v_density_b = torch.log(v_count_b)

                geo_embed = self.geo_embed_layer(torch.cat((v_xyz_b, v_density_b), dim=-1))
                
                # 打印最大值，最小值和平均值，辅助判断分布是否正常
                # print(f"Max Density (Log scale): {v_density_b.max().item():.4f}")
                # print(f"Density Stats -> Min: {v_density_b.min().item():.4f}, Mean: {v_density_b.mean().item():.4f}")
                # voxel 聚合
                v_feat_b = scatter_mean(feat_b, p2v_b, dim=0)  # (num_vox, C)
                v_feat_b = v_feat_b + geo_embed
                batch_features_list.append(v_feat_b)
                batch_counts.append(v_feat_b.shape[0])

            # --- Voxel Self-Attention ---
            # 1. Pad packed features to (B, Max_Voxels, C)
            padded_features = torch.nn.utils.rnn.pad_sequence(batch_features_list, batch_first=True)
            B_pad, L_pad, C_pad = padded_features.shape

            # 2. Create padding mask (True where padded)
            key_padding_mask = torch.zeros((B_pad, L_pad), dtype=torch.bool, device=padded_features.device)
            # Handle case where all batches might be empty (though unlikely)
            if L_pad > 0:
                for i, count in enumerate(batch_counts):
                    key_padding_mask[i, count:] = True
        
            refined_features = self.voxel_self_attn(padded_features, src_key_padding_mask=key_padding_mask)
            refined_features = refined_features.to(torch.bfloat16)

            # 4. Flatten back to packed sequence
            pooled_video_features = refined_features[~key_padding_mask]
            batch_offset = torch.tensor(batch_counts, device=pooled_video_features.device).cumsum(0).to(torch.int32)


            # pooled_video_features = torch.cat(batch_features_list, dim=0)
            # batch_offset = torch.tensor(batch_counts, device=pooled_video_features.device).cumsum(0).to(torch.int32)
            # # --- 方案1: 3D Heatmap Fusion (Grasp Features) ---
            # grasp_features = None
            # if grasps is not None:
                
            #     grasp_features_list = []
            #     sigma = 0.05 # Bandwidth for Gaussian heatmap (meters)

            #     for b in range(len(video_features_up)):
            #         # 1. Get voxel features and centers for current batch
            #         # p2v[b] maps points to voxel indices
            #         # video_features_up[b]: (N, C)
            #         v_feat = scatter_mean(video_features_up[b], p2v[b], dim=0) # (Nv, C)
                    
            #         # xyz_down[b]: (N, 3)
            #         xyz_flat = xyz_down[b].view(-1, 3)
            #         v_xyz = scatter_mean(xyz_flat, p2v[b], dim=0) # (Nv, 3)

            #         # 2. Compute Distance: Grasp to Voxels
            #         g_xyz = grasps[b] # (25, 6, 3)
                    
            #         # (25, 6, 1, 3) - (1, 1, Nv, 3) -> (25, 6, Nv)
            #         # This computes dist from every grasp keypoint to every voxel center
            #         dists = torch.norm(g_xyz.unsqueeze(2) - v_xyz.unsqueeze(0).unsqueeze(0), p=2, dim=-1)
                    
            #         # Min distance from any of 6 keypoints to voxel
            #         min_dists = dists.min(dim=1)[0] # (25, Nv)

            #         # 3. Gaussian Heatmap Weights
            #         weights = torch.exp(-(min_dists**2) / (sigma**2))
                    
            #         # Normalize attention weights
            #         weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

            #         # 4. Integrate Features
            #         # (25, Nv) @ (Nv, C) -> (25, C)
            #         g_feat_b = torch.mm(weights, v_feat)
            #         grasp_features_list.append(g_feat_b)

            #     grasp_features = torch.stack(grasp_features_list, dim=0)

        
        return pooled_video_features, batch_offset  # (B, num_token, 1024) or (Bn, 1024)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
