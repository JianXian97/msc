from typing import Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks import UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.convolutions import ResidualUnit
from monai.networks.blocks.dynunet_block import get_conv_layer
 
from networks.mobilevit import MobileVitBlock
from networks.cft import CFT
from networks.caupblock import CAUpBlock

class UNETMV(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int] = (16,16,16),
        feature_size: int = 16,
        hidden_size: int = 144,
        mlp_dim: int = 576,
        num_heads: int = 12,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        decode_mode: str = "simple",
        cft_mode: str = "channel"
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
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), norm_name='instance')

        """
        print("Init UNETMV")
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")
            
        if mlp_dim / hidden_size != 4:
            raise AssertionError("mlp dim should be 4 times of hidden size.")
            
        for i, size in enumerate(img_size):
            if size < patch_size[i]:
                raise AssertionError("img size cannot be smaller than patch size for dim " + str(i))
            
            # if patch_size[i] < 16:
            #     raise AssertionError("min patch size is 16 for each dimension")
        
        if decode_mode not in ['simple', 'CA']:
            raise AssertionError("decode mode should be either 'simple' or 'CA'.")
            


        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        
        self.preprocess = ResidualUnit(
            spatial_dims = 3,
            in_channels = in_channels,
            out_channels = feature_size,
            strides = 2,
            kernel_size = 3,
            act = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            norm = "INSTANCE",
            dropout = dropout_rate,
            bias = True,
            last_conv_only = False,
        )
        
        # self.skip = ResidualUnit(
        #     spatial_dims = 3,
        #     in_channels = in_channels,
        #     out_channels = feature_size,
        #     strides = 1,
        #     kernel_size = 3,
        #     act = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        #     norm = "INSTANCE",
        #     dropout = dropout_rate,
        #     bias = True,
        #     last_conv_only = False,
        # )
        
 
        
        self.cft_scale = 1
        self.cft = CFT(
                in_channels = feature_size*(2**self.cft_scale),
                dropout_rate = 0,
                norm_name = "instance",      
                transformer_dim = hidden_size,
                hidden_dim = mlp_dim,
                num_heads = num_heads,
                num_layers = 3,
                img_size = tuple(x // 2**self.cft_scale for x in img_size),   
                patch_size = tuple(x // 2**0 for x in self.patch_size),
                out_channels = feature_size*(2**self.cft_scale),   \
                mode = cft_mode
                )
        
        self.mobilevit_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        img_size_list = [img_size]
        for i in range(5):
            patch_size = tuple(max(x // 2**i,1) for x in self.patch_size)
            img_size = tuple(max(x // 2,1) for x in img_size)
            img_size_list.append(img_size)
            layer = MobileVitBlock(
                in_channels = feature_size * (2 ** i),
                dropout_rate = dropout_rate,
                norm_name = norm_name,
                #transformer params
                transformer_dim = hidden_size,
                hidden_dim = mlp_dim,
                num_heads = num_heads,
                num_layers = 3,
                #fold params
                img_size = img_size,   
                patch_size = patch_size,
                #mobile vit additional params
                out_channels = feature_size * (2 ** (i+1)),        
            )
            # in_channels = feature_size * (2 ** i)
            

            self.mobilevit_blocks.append(layer)
            
        for i in range(4):
            #downsample layers, only 4 needed
            downsample_strides = tuple([i//j for i, j in zip(img_size_list[i+1], img_size_list[i+2])]) #handle sizes when inputs size is < (96,96,96)
            layer = ResidualUnit(
                spatial_dims = 3,
                in_channels = feature_size * (2 ** (i+1)),
                out_channels = feature_size * (2 ** (i+1)),
                strides = downsample_strides,
                kernel_size = 3,
                act = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
                norm = "INSTANCE",
                dropout = dropout_rate,
                bias = True,
                last_conv_only = False,
            )
    
            self.downsample_blocks.append(layer)
        

        self.decoders = nn.ModuleList()
        if decode_mode == "simple":
            #TODO
            
            for i in range(4):
                upsample_strides = tuple([i//j for i, j in zip(img_size_list[i+1], img_size_list[i+2])])
                layer = UnetrUpBlock(
                    spatial_dims=3,
                    in_channels=feature_size * (2**(i+2)),
                    out_channels=feature_size * (2**(i+1)),
                    kernel_size=1,
                    upsample_kernel_size=upsample_strides,
                    norm_name=norm_name,
                    res_block=res_block,
                )
                self.decoders.append(layer)
                
                
        else:#mode == "CA"
            for i in range(4):
                layer = CAUpBlock(
                    spatial_dims = 3,
                    in_channels=feature_size * (2**(i+2)),
                    out_channels=feature_size * (2**(i+1)),
                    kernel_size = 1,
                    upsample_kernel_size = 2,
                    norm_name = norm_name,
                    patch_size = tuple(int(x // 2**(i)) for x in self.patch_size),
                    img_size = tuple(x // 2**(i+1) for x in self.img_size),
                    #transformer params
                    transformer_dim = hidden_size,
                    hidden_dim = mlp_dim,
                    num_heads = num_heads,
                    dropout_rate = dropout_rate,
                )
                self.decoders.append(layer)
            
            
        # self.postprocess = self.decoders[0]
        self.postprocess = get_conv_layer(
                spatial_dims=3,
                in_channels=feature_size*2,
                out_channels=feature_size,
                kernel_size=2,
                stride=2,
                conv_only=True,
                is_transposed=True,
            )
   

 
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore
        
 
        
        
    def forward(self, x_in):
        if not tuple(x_in.shape[2:]) == self.img_size:
            raise AssertionError(f"Input shape is wrong, expected {tuple(x_in.shape[2:])}, received {self.img_size}")
        
        x_in1 = self.preprocess(x_in)
        enc1 = self.mobilevit_blocks[0](x_in1)
        x_in2 = self.downsample_blocks[0](enc1)
        enc2 = self.mobilevit_blocks[1](x_in2)
        x_in3 = self.downsample_blocks[1](enc2)
        enc3 = self.mobilevit_blocks[2](x_in3)
        x_in4 = self.downsample_blocks[2](enc3)
        enc4 = self.mobilevit_blocks[3](x_in4)
        x_in5 = self.downsample_blocks[3](enc4)
        enc5 = self.mobilevit_blocks[4](x_in5)
                
        enc1,enc2,enc3,enc4 = self.cft(enc1, enc2, enc3, enc4)
        
        dec4 = self.decoders[3](enc5, enc4)       
        dec3 = self.decoders[2](dec4, enc3)              
        dec2 = self.decoders[1](dec3, enc2)
        dec1 = self.decoders[0](dec2, enc1)

        # skip = self.skip(x_in)
        # dec1 = self.decoders[0](dec1, skip)
        dec1 = self.postprocess(dec1)
        return self.out(dec1)
