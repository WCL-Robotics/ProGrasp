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
        shape=None, multiview_data=None, voxelize=None,
        ) -> torch.Tensor:
        """
        Args:
            feature_list: list of tensor (B*V, C, H, W)
            xyz_list: list of tensor (B*V, H, W, 3)
            shape: (B, V)
        """
        out_features = []
        bs, v = shape
        for j, (feature, xyz) in enumerate(zip(feature_list, xyz_list)):
            # B*V, F, H, W -> B, V, F, H, W -> B, V, H, W, F
            bv, f, h, w = feature.shape
            
            # feature: (B*V, C, H, W) -> mean -> (B*V, C)
            # xyz: (B*V, H, W, 3) -> mean -> (B*V, 3)
            # f_pool = feature.mean(dim=[2, 3])
            # xyz_pool = xyz.mean(dim=[1, 2])
            # gate_in = torch.cat([f_pool, xyz_pool], dim=-1) # (B*V, C+3)
            # view_weight = (self.gate_net(gate_in)+ 1.0) / 2.0
            # view_weight = view_weight.view(bs, v, 1, 1, 1)

            feature = feature.reshape(bs, v, f, h, w).permute(0, 1, 3, 4, 2)
            xyz = xyz.reshape(bs, v, h, w, 3)
            pos_embed = self.encode_pe(xyz) # (B, V, H, W, F)
            # feature = (feature  + pos_embed) * view_weight
            feature = feature  + pos_embed
            feature = feature.flatten(1, 3)  # (B, V*H*W, F)
            out_features.append(feature)
        return out_features