import torch
import torch.nn as nn

import numpy as np

from typing import Sequence, Tuple, Union
from monai.networks.blocks.mlp import MLPBlock
from networks.crossattention import CABlock
from networks.mobilevit import MobileVitBlock, fold_proj, unfold_proj

from monai.networks.blocks.convolutions import Convolution
torch.manual_seed(0)

class CrossTransformerBlock(nn.Module):     
    def __init__(self, hidden_size: int, mlp_dim: int, num_heads: int, dropout_rate: float = 0.0) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = CABlock(hidden_size, num_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, xq, xkv):
        assert len(xq) == len(xkv), "different input size!"
        _, c1, _ = xq.shape
        _, c2, _ = xkv.shape
        xqkv = self.norm1(torch.cat((xq, xkv), dim = 1))
        xq, xkv = torch.split(xqkv, [c1, c2], dim = 1)
        x = xq + self.attn(xq, xkv)
        x = x + self.mlp(self.norm2(x))
        return x
