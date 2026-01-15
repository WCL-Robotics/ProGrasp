import torch
from torch import nn
from .position_encodings import PositionEmbeddingLearnedMLP
    

class SpatialAwareModule(nn.Module):

    def __init__(self, latent_dim=1024):
        super(SpatialAwareModule, self).__init__()
        self.latent_dim = latent_dim
        self.initialize()

    def initialize(self):
        self.positional_embedding = PositionEmbeddingLearnedMLP(dim=3, num_pos_feats=self.latent_dim)
        
        # Gating Network
        self.gate_net = nn.Sequential(
            nn.Linear(self.latent_dim + 3, self.latent_dim // 4),
            nn.GELU(),
            nn.Linear(self.latent_dim // 4, 1),
            nn.Tanh()
        )

        # print(f"\n[SpatialAwareModule] Initialized parameters:")
        # for name, param in self.named_parameters():
        #     print(f"  {name}: {param.shape}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")


    def encode_pe(self, xyz=None):
        return self.positional_embedding(xyz)
        
    def forward(
        self, feature_list=None, xyz_list=None,
        shape=None, valid_mask_feat=None, multiview_data=None, voxelize=None,
        ) -> torch.Tensor:
        """
        Args:
            feature_list: list of tensor (B*V, C, H, W)
            xyz_list: list of tensor (B*V, H, W, 3)
            shape: (B, V)
            valid_mask_feat: tensor (B*V, H, W, 1)
        """
        out_features = []
        bs, v = shape
        for j, (feature, xyz) in enumerate(zip(feature_list, xyz_list)):
            # B*V, F, H, W -> B, V, F, H, W -> B, V, H, W, F
            bv, f, h, w = feature.shape
            
            # 1. 调整维度以便拼接: (B*V, H, W, C) + (B*V, H, W, 3) -> (B*V, H, W, C+3)
            f_perm = feature.permute(0, 2, 3, 1) # (bv, h, w, f)
            gate_in = torch.cat([f_perm, xyz], dim=-1) # (bv, h, w, f+3)
            
            # 2. 逐像素计算权重 (Projection)
            # (bv, h, w, f+3) -> Linear -> (bv, h, w, 1)
            pixel_weight = (self.gate_net(gate_in) + 1.0) / 2.0
            pixel_weight = pixel_weight * valid_mask_feat
            
            # 3. Reshape 权重以广播到 Feature
            # (bv, h, w, 1) -> (bs, v, h, w, 1)
            pixel_weight = pixel_weight.view(bs, v, h, w, 1)
            valid_mask_feat = valid_mask_feat.view(bs, v, h, w, 1)

            # 4. 特征融合
            feature = feature.reshape(bs, v, f, h, w).permute(0, 1, 3, 4, 2) #(bs, v, h, w, f)
            xyz = xyz.reshape(bs, v, h, w, 3)
            pos_embed = self.encode_pe(xyz) # (B, V, H, W, F)
            
            # 应用逐像素权重
            feature = (feature + pos_embed) * valid_mask_feat
            
            feature = feature.flatten(1, 3)  # (B, V*H*W, F)
            out_features.append(feature)
        return out_features