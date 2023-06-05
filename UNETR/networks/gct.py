import torch
import torch.nn as nn

import numpy as np

from typing import Sequence, Tuple, Union
from monai.networks.blocks.mlp import MLPBlock
from networks.crossattention import CABlock
from networks.mobilevit import MobileVitBlock, fold_proj, unfold_proj

from monai.networks.blocks.convolutions import Convolution
torch.manual_seed(0)
class GCT(MobileVitBlock):
    def __init__ (self,
        #conv params
        in_channels: int = 1,
        strides: Union[Sequence[int], int] = 1,
        dropout_rate: float = 0,
        norm_name: Union[Tuple, str] = "instance",
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        local_out_channels: int = 8,
        #transformer params
        transformer_dim: int = 144,
        hidden_dim: int = 576,
        num_heads: int = 12,
        num_layers: int = 12,
        #fold params
        img_size: Union[Sequence[int], int] = (96, 96, 96),   
        patch_size: Union[Sequence[int], int] = 16,
        #mobile vit additional params
        out_channels: int = 16,        
    )->None : 
        super().__init__(in_channels, strides, dropout_rate, norm_name, act_name, transformer_dim, hidden_dim, num_heads, num_layers, img_size, patch_size, out_channels)
        
        self.transformers = nn.ModuleList(
            [CrossTransformerBlock(transformer_dim, hidden_dim, num_heads, dropout_rate) for i in range(2)]
        )

        self.combine_proj = Convolution(
             self.dimensions,
             2*transformer_dim,
             transformer_dim,
             strides=1,
             kernel_size=1,
             adn_ordering="ADN",
             act=act_name,
             norm=norm_name,
             dropout=0,
             dropout_dim=1,
             dilation=1,
             bias=True,
             conv_only=False,
             padding=0,
        ) 
        
        self.local_out_channels = local_out_channels
        self.local_rep = nn.ModuleList([nn.Sequential(), nn.Sequential(), nn.Sequential()]) #one for each input layer
        self.in_channels = in_channels
        
        num_local_conv_layers = max(max(self.patch_size)//2,1) + 1 #stack layers to increase receptive field. +1 to account for 1x1x1 conv. Min 2 layers
        kernel_sizes = [3] * (num_local_conv_layers - 1) + [1]
        paddings = [1] * (num_local_conv_layers - 1) + [0]
         
        for j in range(3):
            #for different input feature maps with different channel size

            in_channels = self.in_channels * (2**j)
            local_out_channels = self.local_out_channels
            
            for i in range(num_local_conv_layers):
                conv = Convolution(
                    self.dimensions,
                    in_channels,
                    local_out_channels,
                    strides=strides,
                    kernel_size=kernel_sizes[i],
                    adn_ordering="ADN",
                    act=act_name,
                    norm=norm_name,
                    dropout=dropout_rate,
                    dropout_dim=1,
                    dilation=1,
                    bias=True,
                    conv_only=False,
                    padding=paddings[i],
                )
                
                in_channels = local_out_channels
                if i == num_local_conv_layers - 2:
                    local_out_channels = transformer_dim
                
                
                self.local_rep[j].add_module(f"conv{i}", conv)
            
        self.unfold_proj_layer = nn.ModuleList()
        for i in range(3):
            layer = Convolution(
             2,
             np.prod(patch_size),
             transformer_dim,
             strides=1,
             kernel_size=1,
             adn_ordering="ADN",
             act=act_name,
             norm=norm_name,
             dropout=0,
             dropout_dim=1,
             dilation=1,
             bias=True,
             conv_only=False,
             padding=0,
            ) 
            self.unfold_proj_layer.append(layer)
            
          
    def forward(self, f2, f3, f4):
        res = f2
        f2 = self.local_rep[0](f2)
        f3 = self.local_rep[1](f3)
        f4 = self.local_rep[2](f4)
        
        f2 = unfold_proj(f2, self.patch_size, self.unfold_proj_layer[0])
        f3 = unfold_proj(f3, self.patch_size, self.unfold_proj_layer[1])
        f4 = unfold_proj(f4, self.patch_size, self.unfold_proj_layer[2])

        x1 = self.transformers[0](f2, f3)
        x2 = self.transformers[1](f2, f4)
        
        x1 = fold_proj(x1, self.img_size, self.patch_size, self.fold_proj_layer, self.transformer_dim)
        x2 = fold_proj(x2, self.img_size, self.patch_size, self.fold_proj_layer, self.transformer_dim)
        
        x = torch.cat([x1, x2], dim=1)
 
        x = self.combine_proj(x) 
        x = self.fusion(res, x)
        return x

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
