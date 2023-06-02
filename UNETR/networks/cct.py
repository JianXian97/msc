import torch
import torch.nn as nn

import numpy as np

from typing import Sequence, Tuple, Union
from monai.networks.blocks.mlp import MLPBlock
from networks.crossattention import CABlock
from networks.mobilevit import MobileVitBlock

from monai.networks.blocks.convolutions import Convolution
torch.manual_seed(0)

class CCT(MobileVitBlock):
    def __init__ (self,
        #conv params
        in_channels: int = 1,
        strides: Union[Sequence[int], int] = 1,
        dropout_rate: float = 0,
        norm_name: Union[Tuple, str] = "instance",
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
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
        
        
        self.patch_size = np.array(self.patch_size)
        self.img_size = np.array(self.img_size)
        
        cross_trans_dim = np.prod(self.img_size) // np.prod(self.patch_size)
        self.transformers = nn.ModuleList(
            [CrossTransformerBlock(cross_trans_dim, cross_trans_dim*4, num_heads, dropout_rate) for i in range(4)]
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
        
        del self.local_rep #inherited from mobilevit
        self.in_channels = in_channels
        self.unfold_proj_layer = nn.ModuleList()
        self.fold_proj_layer = nn.ModuleList()
        self.fusion_layer = nn.ModuleList()
        for i in range(4):
            layer = Convolution(
             2,
             np.prod(patch_size)//2**(i*3),
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
            
            layer = Convolution(
             2,
             transformer_dim,
             np.prod(patch_size)//2**(i*3),             
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
            self.fold_proj_layer.append(layer)
  
            fusion_channels = out_channels * 2**i            
            layer = Convolution(
                    self.dimensions,
                    2 * fusion_channels,
                    fusion_channels,
                    strides=strides,
                    kernel_size=1,
                    adn_ordering="ADN",
                    act=act_name,
                    norm=norm_name,
                    dropout=dropout_rate,
                    dropout_dim=1,
                    dilation=1,
                    bias=True,
                    conv_only=False,
                    padding=0,
                ) 
            self.fusion_layer.append(layer)
          
    def forward(self, f1, f2, f3, f4):
        res1, res2, res3, res4 = f1, f2, f3, f4
        
        f1 = self.unfold_proj(f1, self.patch_size, self.unfold_proj_layer[0]) 
        f2 = self.unfold_proj(f2, self.patch_size//2, self.unfold_proj_layer[1])
        f3 = self.unfold_proj(f3, self.patch_size//4, self.unfold_proj_layer[2])
        f4 = self.unfold_proj(f4, self.patch_size//8, self.unfold_proj_layer[3])
        
        #channel wise
        f1 = torch.permute(f1, (0,2,1)).contiguous()
        f2 = torch.permute(f2, (0,2,1)).contiguous()
        f3 = torch.permute(f3, (0,2,1)).contiguous()
        f4 = torch.permute(f4, (0,2,1)).contiguous()

        f5 = torch.cat([f1, f2, f3, f4], dim=1)
        
        #channel wise cross attention
        x1 = self.transformers[0](f1, f5)
        x2 = self.transformers[1](f2, f5)
        x3 = self.transformers[2](f3, f5)
        x4 = self.transformers[3](f4, f5)
        
        x1 = torch.permute(x1, (0,2,1)).contiguous()
        x2 = torch.permute(x2, (0,2,1)).contiguous()
        x3 = torch.permute(x3, (0,2,1)).contiguous()
        x4 = torch.permute(x4, (0,2,1)).contiguous()
        
        x1 = self.fold_proj(x1, self.img_size, self.patch_size, self.fold_proj_layer[0])
        x2 = self.fold_proj(x2, self.img_size//2, self.patch_size//2, self.fold_proj_layer[1])
        x3 = self.fold_proj(x3, self.img_size//4, self.patch_size//4, self.fold_proj_layer[2])
        x4 = self.fold_proj(x4, self.img_size//8, self.patch_size//8, self.fold_proj_layer[3])
        
        x1 = self.fusion_layer[0](torch.cat([res1, x1], dim=1))
        x2 = self.fusion_layer[1](torch.cat([res2, x2], dim=1))
        x3 = self.fusion_layer[2](torch.cat([res3, x3], dim=1))
        x4 = self.fusion_layer[3](torch.cat([res4, x4], dim=1))
        
        return x1, x2, x3, x4

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
