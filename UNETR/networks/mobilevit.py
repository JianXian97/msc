import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from typing import Sequence, Tuple, Union

from einops.layers.torch import Rearrange

from monai.utils import ensure_tuple_rep
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.transformerblock import TransformerBlock

torch.manual_seed(0)

class MobileVitBlock(nn.Module):
    '''
    This class defines the MobileViT block for 3D images by generalising the 2D MobileViT block.
        
    '''

    def __init__(
        self,
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
        super().__init__()
        self.dimensions = 3
        self.img_size = ensure_tuple_rep(img_size, self.dimensions)
        self.patch_size = ensure_tuple_rep(patch_size, self.dimensions)
        self.patch_dim = in_channels * np.prod(patch_size) 
       
        self.local_rep = nn.Sequential()
        num_local_conv_layers = self.patch_size[0]//2 + 1 #stack layers to increase receptive field. +1 to account for 1x1x1 conv
        kernel_sizes = [3] * (num_local_conv_layers - 1) + [1]
        paddings = [1] * (num_local_conv_layers - 1) + [0]
        local_out_channels = in_channels
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
            if i == num_local_conv_layers - 2:
                local_out_channels = transformer_dim
            self.local_rep.add_module(f"conv{i}", conv)
        
        #project down from transformer channels to input channels using 1x1x1 conv
        self.conv_proj = Convolution(
                self.dimensions,
                transformer_dim,
                in_channels,
                strides=strides,
                kernel_size=kernel_sizes[-1],
                adn_ordering="ADN",
                act=act_name,
                norm=norm_name,
                dropout=dropout_rate,
                dropout_dim=1,
                dilation=1,
                bias=True,
                conv_only=False,
                padding=paddings[-1],
            ) 
        
        #fusion layer, combine transformer output with input patch
        self.fusion_layer = Convolution(
                self.dimensions,
                2 * in_channels,
                out_channels,
                strides=strides,
                kernel_size=kernel_sizes[0],
                adn_ordering="ADN",
                act=act_name,
                norm=norm_name,
                dropout=dropout_rate,
                dropout_dim=1,
                dilation=1,
                bias=True,
                conv_only=False,
                padding=paddings[0],
            ) 

        self.transformers = nn.ModuleList(
            [TransformerBlock(transformer_dim, hidden_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )


    def unfold(self, x):
        '''
        This function is used to unfold the input image into patches.
        '''
        '''
        b = x.shape[0] #batch size
        unfolded_x = x.unfold(2, self.patch_size[0], self.patch_size[0]).unfold(3, self.patch_size[1], self.patch_size[1]).unfold(4, self.patch_size[2], self.patch_size[2])
        unfolded_x = unfolded_x.reshape(b, -1, self.patch_size[0] * self.patch_size[1] * self.patch_size[2])

        '''
        chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:self.dimensions]
        from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
        to_chars = f"(b {' '.join([c[1] for c in chars])}) ({' '.join([c[0] for c in chars])}) c"
        axes_len = {f"p{i+1}": p for i, p in enumerate(self.patch_size)}
        unfolded = Rearrange(f"{from_chars} -> {to_chars}", **axes_len)(x)
        
        return unfolded
    
    def fold(self, x):
        '''
        This function is used to fold the transformer's output embeddings into the output image.
        '''
        chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:self.dimensions]
        
        from_chars = f"(b {' '.join([c[1] for c in chars])}) ({' '.join([c[0] for c in chars])}) c"
        to_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)

        axes_len = {f"p{i+1}": p for i, p in enumerate(self.patch_size)}
        num_per_axis = {axis[0] : self.img_size[i]//self.patch_size[i] for i, axis in enumerate(chars)}
        folded = Rearrange(f"{from_chars} -> {to_chars}", **axes_len, **num_per_axis)(x) 

        return folded
    
    def fusion(self, img, x):
        '''
        This function is used to combine the transformer output with the input patch.
        '''
        x = self.conv_proj(x)
        x = torch.cat([img, x], dim=1)
        x = self.fusion_layer(x)

        return x
        
    def forward(self, x):
        res = x       
        out = self.local_rep(x)
        out = self.unfold(out)
        for transformer_layer in self.transformers:
            out = transformer_layer(out)    
        out = self.fold(out)
        out = self.fusion(res, out)
        return out