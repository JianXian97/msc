# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:34:27 2023

@author: sooji
"""

from typing import Sequence, Tuple, Union

import numpy as np

import torch
import torch.nn as nn

from monai.utils import ensure_tuple_rep
from monai.networks.blocks.dynunet_block import UnetResBlock, get_conv_layer
from monai.networks.blocks.convolutions import Convolution

from networks.crossattention import CABlock
from networks.mobilevit import fold_proj, unfold_proj
from networks.gct import CrossTransformerBlock

#cross attention upsampling block
class CAUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        patch_size: Union[Sequence[int], int],
        img_size: Union[Sequence[int], int],
        #transformer params
        transformer_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float,
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        self.dimensions = 3
        self.img_size = ensure_tuple_rep(img_size, self.dimensions)
        self.patch_size = ensure_tuple_rep(patch_size, self.dimensions)
        self.transformer_dim = transformer_dim
        
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.conv_block = UnetResBlock(
            spatial_dims,
            transformer_dim,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )
        
        
        self.transformer_proj_layer = nn.ModuleList()
        self.unfold_proj_layer = nn.ModuleList()
        for i in range(2):
            layer = Convolution(
             2,
             np.prod(patch_size),
             transformer_dim,
             strides=1,
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
            self.unfold_proj_layer.append(layer)
            
            layer = Convolution(
             3,
             in_channels//2,
             transformer_dim,
             strides=1,
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
            self.transformer_proj_layer.append(layer)
            
        self.fold_proj_layer = Convolution(
             2,
             transformer_dim,
             np.prod(patch_size),             
             strides=1,
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

        self.transformer = CrossTransformerBlock(transformer_dim, hidden_dim, num_heads, dropout_rate)

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        
        out = self.transformer_proj_layer[0](out)
        skip = self.transformer_proj_layer[1](skip)
        
        out = unfold_proj(out, self.patch_size, self.unfold_proj_layer[0])
        skip = unfold_proj(skip, self.patch_size, self.unfold_proj_layer[1])
       
        out = self.transformer(skip, out)
        
        out = fold_proj(out, self.img_size, self.patch_size, self.fold_proj_layer, self.transformer_dim)
        

        out = self.conv_block(out)
        return out