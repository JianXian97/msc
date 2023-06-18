import torch
import torch.nn as nn

import numpy as np

from typing import Sequence, Tuple, Union
from monai.networks.blocks import ADN
from monai.networks.blocks.mlp import MLPBlock
from networks.crossattention import CABlock
from networks.mobilevit import MobileVitBlock, fold_proj, unfold_proj
from networks.gct import CrossTransformerBlock


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
        self.channel_transformers = nn.ModuleList(
            [CrossTransformerBlock(cross_trans_dim, cross_trans_dim*4, num_heads, dropout_rate) for i in range(4)]
        )
        self.patch_transformers = nn.ModuleList(
            [CrossTransformerBlock(transformer_dim, hidden_dim, num_heads, dropout_rate) for i in range(4)]
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
             dropout=dropout_rate,
             dropout_dim=1,
             dilation=1,
             bias=True,
             conv_only=False,
             padding=0,
        ) 
        
        del self.local_rep #inherited from mobilevit
        del self.transformers #inherited from mobilevit
        self.in_channels = in_channels
        
        self.unfold_proj_layer = nn.ModuleList()
        self.fold_proj_layer = nn.ModuleList()
        self.fusion_layer = nn.ModuleList()
        self.global_proj_layers = nn.ModuleList()
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
             dropout=dropout_rate,
             dropout_dim=1,
             dilation=1,
             bias=True,
             conv_only=False,
             padding=0,
            ) 
            self.unfold_proj_layer.append(layer)
            
            layer = Convolution(
             2,
             np.prod(patch_size)//2**(i*3),
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
             2,
             transformer_dim,
             np.prod(patch_size)//2**(i*3),             
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
            self.fold_proj_layer.append(layer)
            
            layer = Convolution(
             2,
             transformer_dim,
             np.prod(patch_size)//2**(i*3),             
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
            self.fold_proj_layer.append(layer)
                                    
            layer1 = Convolution(
                        self.dimensions,
                        in_channels * (2**i),
                        transformer_dim,
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
            layer2 = Convolution(
                        self.dimensions,
                        transformer_dim,
                        in_channels * (2**i),
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
            
            self.global_proj_layers.append(layer1)
            self.global_proj_layers.append(layer2)
            
  
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

        
        
    def patch_attn(self, x1, x2, x3, x4):
        x1 = self.global_proj_layers[0](x1)
        x2 = self.global_proj_layers[2](x2)
        x3 = self.global_proj_layers[4](x3)
        x4 = self.global_proj_layers[6](x4)
        
        x1 = unfold_proj(x1, self.patch_size, self.unfold_proj_layer[0]) 
        x2 = unfold_proj(x2, self.patch_size//2, self.unfold_proj_layer[2])
        x3 = unfold_proj(x3, self.patch_size//4, self.unfold_proj_layer[4])
        x4 = unfold_proj(x4, self.patch_size//8, self.unfold_proj_layer[6]) 
        x5 = torch.cat([x1, x2, x3, x4], dim=1) 
        
        x1 = self.patch_transformers[0](x1, x5)
        x2 = self.patch_transformers[1](x2, x5)
        x3 = self.patch_transformers[2](x3, x5)
        x4 = self.patch_transformers[3](x4, x5)
        
        x1 = fold_proj(x1, self.img_size, self.patch_size, self.fold_proj_layer[0], self.transformer_dim)
        x2 = fold_proj(x2, self.img_size//2, self.patch_size//2, self.fold_proj_layer[2], self.transformer_dim)
        x3 = fold_proj(x3, self.img_size//4, self.patch_size//4, self.fold_proj_layer[4], self.transformer_dim)
        x4 = fold_proj(x4, self.img_size//8, self.patch_size//8, self.fold_proj_layer[6], self.transformer_dim)
        
        x1 = self.global_proj_layers[1](x1)
        x2 = self.global_proj_layers[3](x2)
        x3 = self.global_proj_layers[5](x3)
        x4 = self.global_proj_layers[7](x4)
        
        return x1, x2, x3, x4
    
    def channel_attn(self, f1, f2, f3, f4):                
        f1 = unfold_proj(f1, self.patch_size, self.unfold_proj_layer[1]) 
        f2 = unfold_proj(f2, self.patch_size//2, self.unfold_proj_layer[3])
        f3 = unfold_proj(f3, self.patch_size//4, self.unfold_proj_layer[5])
        f4 = unfold_proj(f4, self.patch_size//8, self.unfold_proj_layer[7])  
        
        #channel wise
        f1 = torch.permute(f1, (0,2,1)).contiguous()
        f2 = torch.permute(f2, (0,2,1)).contiguous()
        f3 = torch.permute(f3, (0,2,1)).contiguous()
        f4 = torch.permute(f4, (0,2,1)).contiguous()
        f5 = torch.cat([f1, f2, f3, f4], dim=1)
        
        #channel wise cross attention
        f1 = self.channel_transformers[0](f1, f5)
        f2 = self.channel_transformers[1](f2, f5)
        f3 = self.channel_transformers[2](f3, f5)
        f4 = self.channel_transformers[3](f4, f5)
        
        f1 = torch.permute(f1, (0,2,1)).contiguous()
        f2 = torch.permute(f2, (0,2,1)).contiguous()
        f3 = torch.permute(f3, (0,2,1)).contiguous()
        f4 = torch.permute(f4, (0,2,1)).contiguous()
        
        f1 = fold_proj(f1, self.img_size, self.patch_size, self.fold_proj_layer[1], self.transformer_dim)
        f2 = fold_proj(f2, self.img_size//2, self.patch_size//2, self.fold_proj_layer[3], self.transformer_dim)
        f3 = fold_proj(f3, self.img_size//4, self.patch_size//4, self.fold_proj_layer[5], self.transformer_dim)
        f4 = fold_proj(f4, self.img_size//8, self.patch_size//8, self.fold_proj_layer[7], self.transformer_dim)
        
        return f1, f2, f3, f4
        
    def forward(self, f1, f2, f3, f4):
        res1, res2, res3, res4 = f1, f2, f3, f4
        f1, f2, f3, f4 = self.channel_attn(f1, f2, f3, f4)
        # f1, f2, f3, f4 = self.patch_attn(f1, f2, f3, f4)
        

        x1 = self.fusion_layer[0](torch.cat([res1, f1], dim=1))
        x2 = self.fusion_layer[1](torch.cat([res2, f2], dim=1))
        x3 = self.fusion_layer[2](torch.cat([res3, f3], dim=1))
        x4 = self.fusion_layer[3](torch.cat([res4, f4], dim=1))
        
        return x1, x2, x3, x4
