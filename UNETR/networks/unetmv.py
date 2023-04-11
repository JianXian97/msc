# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks import UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock

from monai.networks.blocks.convolutions import ResidualUnit
 
from networks.mobilevit import MobileVitBlock


class UNETMV(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """
        print("Init UNETMV")
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

         
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.mobilevit_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        for i in range(4):
            layer = MobileVitBlock(
                in_channels = in_channels,
                dropout_rate = dropout_rate,
                norm_name = norm_name,
                #transformer params
                transformer_dim = hidden_size,
                hidden_dim = mlp_dim,
                num_heads = num_heads,
                num_layers = 3,
                #fold params
                img_size = img_size,   
                patch_size = self.patch_size,
                #mobile vit additional params
                out_channels = feature_size * (2 ** i),        
            )
            in_channels = feature_size * (2 ** i)
            img_size = tuple(x // 2 for x in img_size)
            self.patch_size = tuple(x // 2 for x in self.patch_size)

            self.mobilevit_blocks.append(layer)
            if i == 3:
                break

            #downsample layers, only 3 needed
            layer = ResidualUnit(
                spatial_dims = 3,
                in_channels = feature_size * (2 ** i),
                out_channels = feature_size * (2 ** i),
                strides = 2,
                kernel_size = 3,
                act = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
                norm = "INSTANCE",
                dropout = dropout_rate,
                bias = True,
                last_conv_only = False,
            )

            self.downsample_blocks.append(layer)
        
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore
  
    def forward(self, x_in):
        enc1 = self.mobilevit_blocks[0](x_in)
        x_in2 = self.downsample_blocks[0](enc1)
        enc2 = self.mobilevit_blocks[0](x_in2)
        x_in3 = self.downsample_blocks[0](enc2)
        enc3 = self.mobilevit_blocks[0](x_in3)
        x_in4 = self.downsample_blocks[0](enc3)
        enc4 = self.mobilevit_blocks[0](x_in4)
 
        dec3 = self.decoder3(enc3, enc4)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        out = self.out(dec1)
        return out
