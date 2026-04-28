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
            valid_mask_feat = valid_mask_feat.view(bs, v, h, w, 1)

            feature = feature.reshape(bs, v, f, h, w).permute(0, 1, 3, 4, 2) #(bs, v, h, w, f)
            xyz = xyz.reshape(bs, v, h, w, 3)
            pos_embed = self.encode_pe(xyz) # (B, V, H, W, F)

            feature = (feature + pos_embed) * valid_mask_feat
            
            feature = feature.flatten(1, 3)  # (B, V*H*W, F)
            out_features.append(feature)
        return out_features