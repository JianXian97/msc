import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from typing import Sequence, Tuple, Union

from einops.layers.torch import Rearrange

from monai.utils import ensure_tuple_rep
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.layers.utils import get_act_layer


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
        self.transformer_dim = transformer_dim 
        # self.patch_dim = in_channels * np.prod(patch_size) 
       
        self.local_rep = nn.Sequential()
        num_local_conv_layers = max(max(self.patch_size)//2,1) + 1 #stack layers to increase receptive field. +1 to account for 1x1x1 conv. Min 2 layers
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
           
        #fusion layer, combine transformer output with input patch using 1x1x1 conv
        self.fusion_layer = Convolution(
                self.dimensions,
                2 * in_channels,
                out_channels,
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
        

        #axis dims = Input's dim / Patch's dim
        self.axis_dims = [i//j for i, j in zip(self.img_size, self.patch_size)]
        self.transformers = nn.ModuleList(
            [TransformerBlock(transformer_dim, hidden_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        
        self.proj_patch_size = [4, 4, 4]
        
        patch_total_size = np.prod(self.patch_size)  
        
        '''
        self.unfold_proj_layer = nn.ModuleList()
        self.fold_proj_layer = nn.ModuleList()
        for i in range(self.dimensions):
            axis_proj = Convolution(
                3,
                self.patch_size[i],
                self.proj_patch_size[i],
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
            self.unfold_proj_layer.append(axis_proj)
            axis_proj = Convolution(
                3,
                self.proj_patch_size[i],
                self.patch_size[i],
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
            self.fold_proj_layer.append(axis_proj)
          '''
          
        self.unfold_proj_layer = Convolution(
             2,
             patch_total_size,
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

        self.fold_proj_layer = Convolution(
            2,
            transformer_dim,
            patch_total_size,
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

    
    def axial_attn(self, x):
        #based off https://github.com/AsukaDaisuki/MAT/blob/main/lib/models/axialnet.py#L161
        chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:self.dimensions]
        axes_len = {f"p{i+1}": p for i, p in enumerate(self.patch_size)} #p1, p2, p3
        num_per_axis = {axis[0] : self.img_size[i]//self.patch_size[i] for i, axis in enumerate(chars)} #h, w, d


        from_chars = "(b t) (h w d) c"
        to_chars = "(b t w d) h c"
        x = Rearrange(f"{from_chars} -> {to_chars}", **num_per_axis, t=self.transformer_dim)(x) #axis 1
        x = self.transformers[0](x)
        
        from_chars = "(b t w d) h c"
        to_chars = "(b t h d) w c"
        x = Rearrange(f"{from_chars} -> {to_chars}", **num_per_axis, t=self.transformer_dim)(x) #axis 2
        x = self.transformers[1](x)
        
        from_chars = "(b t h d) w c"
        to_chars = "(b t h w) d c"
        x = Rearrange(f"{from_chars} -> {to_chars}", **num_per_axis, t=self.transformer_dim)(x) #axis 3
        x = self.transformers[2](x)
        
        from_chars = "(b t h w) d c"
        to_chars =  "(b t) (h w d) c"
        x = Rearrange(f"{from_chars} -> {to_chars}", **num_per_axis, t=self.transformer_dim)(x) 
        
        return x
        
        
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
        x = self.local_rep(x)  
        # res_local = x
        x = unfold_proj(x, self.patch_size, self.unfold_proj_layer)
        torch.cuda.empty_cache()
        # for transformer_layer in self.transformers:
        #     x = transformer_layer(x) 
        x = self.axial_attn(x)
          
        torch.cuda.empty_cache()
        x = fold_proj(x, self.img_size, self.patch_size, self.fold_proj_layer, self.transformer_dim)
        x = self.fusion(res, x)
        return x
 
    
def unfold_proj(x, patch_size, unfold_proj_layer):
    '''
    This function is used to unfold the input image into patches.
    '''
    '''
    b = x.shape[0] #batch size
    unfolded_x = x.unfold(2, self.patch_size[0], self.patch_size[0]).unfold(3, self.patch_size[1], self.patch_size[1]).unfold(4, self.patch_size[2], self.patch_size[2])
    unfolded_x = unfolded_x.reshape(b, -1, self.patch_size[0] * self.patch_size[1] * self.patch_size[2])

    '''
    chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))
    from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
    to_chars = f"b ({' '.join([c[1] for c in chars])}) ({' '.join([c[0] for c in chars])}) c"
    axes_len = {f"p{i+1}": p for i, p in enumerate(patch_size)}
    x = Rearrange(f"{from_chars} -> {to_chars}", **axes_len)(x).contiguous()
    x = unfold_proj_layer(x)
    from_chars = "b z y c"
    to_chars = "(b z) y c"
    x = Rearrange(f"{from_chars} -> {to_chars}")(x).contiguous()
    
    '''
    chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:self.dimensions]
    axes_len = {f"p{i+1}": p for i, p in enumerate(self.patch_size)} #p1, p2, p3
    num_per_axis = {axis[0] : self.img_size[i]//self.patch_size[i] for i, axis in enumerate(chars)} #h, w, d
 
    from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
    to_chars = f"b {' '.join([c[1] for c in chars])} ({' '.join([c[0] for c in chars])} c)"
    x = Rearrange(f"{from_chars} -> {to_chars}", **axes_len)(x) #b p1 p2 p3 (h w d c)
    x = self.unfold_proj_layer[0](x)
    x = torch.permute(x, (0,2,1,3,4)) #b p2 p1 p3 (h w d c)
    x = self.unfold_proj_layer[1](x) 
    x = torch.permute(x, (0,3,2,1,4)) #b p3 p1 p2 (h w d c)
    x = self.unfold_proj_layer[2](x) 
    x = torch.permute(x, (0,2,3,1,4)) #b p1 p2 p3 (h w d c) 
            
    from_chars = f"b {' '.join([c[1] for c in chars])} ({' '.join([c[0] for c in chars])} c)"
    to_chars = f"(b {' '.join([c[1] for c in chars])}) ({' '.join([c[0] for c in chars])}) c"
    x = Rearrange(f"{from_chars} -> {to_chars}", **num_per_axis)(x) #(b p1 p2 p3) (h w d) c
    '''
    
    '''
    from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
    to_chars = f"(b {' '.join([c[1] for c in chars])}) ({' '.join([c[0] for c in chars])}) c"
    x = Rearrange(f"{from_chars} -> {to_chars}", **axes_len)(x) 
    '''
    return x

def fold_proj(x, img_size, patch_size, fold_proj_layer, transformer_dim):
    '''
    This function is used to fold the transformer's output embeddings into the output image.
    '''
    chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))
    axes_len = {f"p{i+1}": p for i, p in enumerate(patch_size)}
    num_per_axis = {axis[0] : img_size[i]//patch_size[i] for i, axis in enumerate(chars)}
 
    from_chars = "(b z) y c"
    to_chars = "b z y c"
    x = Rearrange(f"{from_chars} -> {to_chars}", z = transformer_dim)(x).contiguous()        
    x = fold_proj_layer(x)
    
    from_chars = f"b ({' '.join([c[1] for c in chars])}) ({' '.join([c[0] for c in chars])}) c"
    to_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
    x = Rearrange(f"{from_chars} -> {to_chars}", **axes_len, **num_per_axis)(x).contiguous() 
    
    '''
    chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:self.dimensions]
    axes_len = {f"p{i+1}": p for i, p in enumerate(self.proj_patch_size)} #p1, p2, p3
    num_per_axis = {axis[0] : self.img_size[i]//self.patch_size[i] for i, axis in enumerate(chars)} #h, w, d

   
    from_chars = f"(b {' '.join([c[1] for c in chars])}) z c"
    to_chars = f"b {' '.join([c[1] for c in chars])} (z c)"
    x = Rearrange(f"{from_chars} -> {to_chars}", **axes_len)(x) #b p1 p2 p3 (h w d c)       
    x = self.fold_proj_layer[0](x)
    x = torch.permute(x, (0,2,1,3,4)) #b p2 p1 p3 (h w d c)
    x = self.fold_proj_layer[1](x) 
    x = torch.permute(x, (0,3,2,1,4)) #b p3 p1 p2 (h w d c)
    x = self.fold_proj_layer[2](x) 
    x = torch.permute(x, (0,2,3,1,4)) #b p1 p2 p3 (h w d c) 
    
    from_chars = f"b {' '.join([c[1] for c in chars])} ({' '.join([c[0] for c in chars])} c)"
    to_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
    x = Rearrange(f"{from_chars} -> {to_chars}", **num_per_axis)(x) 
    '''
    
    '''        
    chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:self.dimensions]
    axes_len = {f"p{i+1}": p for i, p in enumerate(self.patch_size)} #p1, p2, p3
    num_per_axis = {axis[0] : self.img_size[i]//self.patch_size[i] for i, axis in enumerate(chars)} #h, w, d

    from_chars = f"(b {' '.join([c[1] for c in chars])}) ({' '.join([c[0] for c in chars])}) c"
    to_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
    x = Rearrange(f"{from_chars} -> {to_chars}", **axes_len)(x) 
    '''
    
    return x        