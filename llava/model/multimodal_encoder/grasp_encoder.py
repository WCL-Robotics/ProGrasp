import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.autograd import Function
from torch.nn.modules.utils import _pair

from mmdet3d.registry import MODELS
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape, dynamic_scatter_3d
from .unproject import voxelize as voxelize_for_scatter

class GraspNet(nn.Module):
    """
    Class to encoder the grasping pose.
    """
    def __init__(self):
        super(GraspNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096)
        )
    
    def forward(self, g):
        return self.net(g)

