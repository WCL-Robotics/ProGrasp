import torch
import torch.nn as nn

from .video_processor import RGBDVideoProcessor
from .spatial_aware_module import SpatialAwareModule
from .unproject import backprojector_dataloader, voxelize, interpolate_feat_up, interpolate_xyz_down
from pytorch3d.ops import sample_farthest_points
from torch_scatter import scatter_mean
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
        self.voxel_size = 0.025
        self.vision_tower_name = vision_tower
        self.video_tower_name = video_tower

        self.pc_range = [-0.5, -0.5, 0, 0.5, 0.5, 1]

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_video_tower', False):
            self.load_model()
        else:
            self.cfg_only = None

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


    def forward(self, features, depths, poses, intrinsics, lengths=None, grasps=None):
        """
        Compute visual features/position embeddings for each patch.

        Args:
            - features: (B, V, 1024, 336, 336), image token features
            - depths: (B, V, H, W), depth images
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

        # (B, V*H*W, C)
        video_features = self.video_tower([features.flatten(0, 1)], [feat_xyz.flatten(0, 1)], (B, V))[0]
        video_features = video_features.reshape(B*V, H, W, C).contiguous()
        video_features_up = interpolate_feat_up(video_features.clone()).reshape(B, -1, C)
        xyz_down = interpolate_xyz_down(xyz.clone())

        # print("xyz_down[..., 0] min:", xyz_down[..., 0].min().item(), "max:", xyz_down[..., 0].max().item())
        # print("xyz_down[..., 1] min:", xyz_down[..., 1].min().item(), "max:", xyz_down[..., 1].max().item())
        # print("xyz_down[..., 2] min:", xyz_down[..., 2].min().item(), "max:", xyz_down[..., 2].max().item())

        valid_inds = (xyz_down[..., 0] > self.pc_range[0]) & (xyz_down[..., 0] < self.pc_range[3]) & \
            (xyz_down[..., 1] > self.pc_range[1]) & (xyz_down[..., 1] < self.pc_range[4]) & \
                (xyz_down[..., 2] > self.pc_range[2]) & (xyz_down[..., 2] < self.pc_range[5])


        xyz_down[~valid_inds] = xyz_down[~valid_inds] * 0
        video_features_up[~valid_inds.view(B, -1)] = video_features_up[~valid_inds.view(B, -1)] * 0

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
            p2v = voxelize(xyz_down, self.voxel_size)  # （B, N)
            pooled_video_features = torch.cat([scatter_mean(video_features_up[b], p2v[b], dim=0) for b in range(len(video_features_up))]) # bn, F
            batch_offset = ((p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
            
            # --- 方案1: 3D Heatmap Fusion (Grasp Features) ---
            grasp_features = None
            if grasps is not None:
                
                grasp_features_list = []
                sigma = 0.05 # Bandwidth for Gaussian heatmap (meters)

                for b in range(len(video_features_up)):
                    # 1. Get voxel features and centers for current batch
                    # p2v[b] maps points to voxel indices
                    # video_features_up[b]: (N, C)
                    v_feat = scatter_mean(video_features_up[b], p2v[b], dim=0) # (Nv, C)
                    
                    # xyz_down[b]: (N, 3)
                    xyz_flat = xyz_down[b].view(-1, 3)
                    v_xyz = scatter_mean(xyz_flat, p2v[b], dim=0) # (Nv, 3)

                    # 2. Compute Distance: Grasp to Voxels
                    g_xyz = grasps[b] # (25, 6, 3)
                    
                    # (25, 6, 1, 3) - (1, 1, Nv, 3) -> (25, 6, Nv)
                    # This computes dist from every grasp keypoint to every voxel center
                    dists = torch.norm(g_xyz.unsqueeze(2) - v_xyz.unsqueeze(0).unsqueeze(0), p=2, dim=-1)
                    
                    # Min distance from any of 6 keypoints to voxel
                    min_dists = dists.min(dim=1)[0] # (25, Nv)

                    # 3. Gaussian Heatmap Weights
                    weights = torch.exp(-(min_dists**2) / (sigma**2))
                    
                    # Normalize attention weights
                    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

                    # 4. Integrate Features
                    # (25, Nv) @ (Nv, C) -> (25, C)
                    g_feat_b = torch.mm(weights, v_feat)
                    grasp_features_list.append(g_feat_b)

                grasp_features = torch.stack(grasp_features_list, dim=0)

        else:
            raise NotImplementedError
        
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
