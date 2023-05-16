import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import roi_align, batched_nms

import numpy as np

from monai.utils import optional_import
from typing import Sequence, Tuple, Union

from networks.mobilevit import MobileVitBlock

from einops.layers.torch import Rearrange

from utils.utils import nms

einops, _ = optional_import("einops")

class BLT(nn.Module):
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

        self.patch_size = patch_size
        self.img_size = img_size

        self.mobilevit = MobileVitBlock(in_channels, strides, dropout_rate, norm_name, act_name, transformer_dim, hidden_dim, num_heads, num_layers, img_size, patch_size, out_channels)

        self.windows = self.Shifted_Windows() #coordinates of all possible windows, M x 6
        self.num_windows = np.prod(self.img_size) // np.prod(self.patch_size) #number of windows to be selected
 
        self.weights = nn.Parameter(torch.ones(1, 1, *self.patch_size))
        
    def calc_win_scores(self, x):
        #x is the feature map
        out = F.conv3d(x, self.weights, bias=None, stride=1, padding=0)/np.prod(self.patch_size) #average 
        return out

    def Shifted_Windows(self, stride=2):
        #adapted from https://github.com/xianlin7/BATFormer/blob/main/models/transformer_parts_mp.py#L34
        #stride = 2 as the feature map's resolution is twice of GCT's output
        #output = windows of shape (M, 6) where M is the number of windows, 6 is the coordinates of the window

        w,h,d = self.img_size
        shift_x = torch.arange(0, w-self.patch_size[0]+1, stride)   
        shift_y = torch.arange(0, h-self.patch_size[1]+1, stride)
        shift_z = torch.arange(0, d-self.patch_size[2]+1, stride)
         
        shift_x, shift_y, shift_z = np.meshgrid(shift_x, shift_y, shift_z)

        shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_z.ravel(), shift_x.ravel(), shift_y.ravel(), shift_z.ravel()), axis=1)
        M = shift.shape[0]
        window = shift.reshape(M, 6)
        window[:, 3:] = self.patch_size
 
        return torch.tensor(window)
    
    def calc_entropy(self, prob):
        #channels = num of classes, located at dim 1
        log_prob = torch.log2(prob + 1e-10)
        entropy = -1 * torch.sum(prob * log_prob, dim=1)  
        return entropy.unsqueeze(1)
    
    def forward(self, x, p_gct):
        #p_gct should be the output of the previous GCT block, channels = num of classes. 
        b, c, h, w, d = x.shape

        entropy = self.calc_entropy(p_gct)

        window_scores = self.calc_win_scores(entropy).view(b, -1) #b x M, M is the total number of all possible windows

        # x_unfold = x.unfold(2,self.patch_size[0],1).unfold(3,self.patch_size[1],1).unfold(4,self.patch_size[2],1)
        # x_unfold = x_unfold.permute(0,1,3,2,4,5,6,7).reshape(b,c,-1,*self.patch_size)
        x1 = x.permute(0,1,3,2,4)
        x1 = x1.unfold(3,self.patch_size[0],1).unfold(2,self.patch_size[1],1).unfold(4,self.patch_size[2],1)
        _,_,h1,w1,d1,_,_,_ = x1.shape
        unfold_shape = (h1, w1, d1)
        # x1 = x1.reshape(b,c,-1,*self.patch_size)
        
        x_selected = torch.zeros((b, self.num_windows, *self.patch_size)) #b x num_windows x prod(patch_size)

        for i in range(b):
            selected_windows_indices = nms(self.windows, window_scores[i,:], iou_threshold=0.2)[:self.num_windows] #b x num_windows, indices of the selected windows
            h, w, d = np.unravel_index(selected_windows_indices, unfold_shape)
            x_selected = x1[i, :, selected_windows_indices, :, :, :] 
 
        from_chars = "b c (h w d) p1 p2 p3"
        to_chars = "b c (h p1) (w p2) (d p3)"
        x_selected = Rearrange(f"{from_chars} -> {to_chars}", h=h, w=w, d=d)(x_selected) 
        x_selected = self.mobilevit(x_selected)
        return x + x_selected
    
