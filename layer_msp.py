import mindspore
import mindspore.nn as nn
import mindspore.ops.functional as F
from ResizeBilinear import *

# A net similar to hourglass, used in the UV net (Location Net)
class HgNet(nn.Cell):
    def __init__(self,in_channels,level,nl_layer=nn.ReLU(),norm_type="GN"):
        super(HgNet,self).__init__()

        down_layers = []
        up_layers = []
        if norm_type = "GN":
            self.norm = nn.GroupNorm(in_channles//8,in_channels)
        elif norm_type ="BN":
            affine=True
            self.norm = nn.BatchNorm2d(in_channels,affine=affine)

        for i in range(level):
            out_channels = in_channels*2
            down_layers.append(
                nn.SequentialCell(
                    ConvBottleNeck(in_channels=in_channels,out_channels=out_channels,nl_layer=nl_layer,norm_type=norm_type),
                    nn.MaxPool2d(kernel_size=2,stride=2)
                    )
                )
            up_layers.append(
                nn.SequentialCell(
                    ResizeBilinear(scale_factor=2),
                    ConvBottleNeck(in_channels=out_channels,out_channels=in_channels,nl_layer=nl_layer,norm_type=norm_type)
                    )
                )
            in_channels=out_channels
        self.down_layers=nn.CellList(down_layers)
        self.up_layers=nn.CellList(up_layers)
    def construct(self,x):
        feature_list=[]
        y=x
        for i in range(len(self.down_layers)):
            feature_list.append(y)
            feature=self.down_layers[i](y)
        for i in range(len(self.down_layers)-1,-1,-1):
            y=self.up_layers[i](y)+feature_list[i]
        y=self.norm(y)
        return y


class ConvBottleNeck(nn.Cell):

    def __init__(self,in_channels,out_channels,nl_layer=nn.ReLU(), norm_type='GN'):
        super(ConvBottleNeck,self).__init__()
        self.nl_layer = nl_layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)

        if norm_type == 'BN':
            affine = True
            # affine = False
            self.norm1 = nn.BatchNorm2d(out_channels // 2, affine=affine)
            self.norm2 = nn.BatchNorm2d(out_channels // 2, affine=affine)
            self.norm3 = nn.BatchNorm2d(out_channels, affine=affine)
        elif norm_type == 'SYBN':
            affine = True
            # affine = False
            self.norm1 = nn.SyncBatchNorm(out_channels // 2, affine=affine)
            self.norm2 = nn.SyncBatchNorm(out_channels // 2, affine=affine)
            self.norm3 = nn.SyncBatchNorm(out_channels, affine=affine)
        else:
            self.norm1 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))
            self.norm2 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))
            self.norm3 = nn.GroupNorm(out_channels // 8, out_channels)

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
    def construct(self, x):

        residual = x

        y = self.conv1(x)
        y = self.norm1(y)
        y = self.nl_layer(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = self.nl_layer(y)

        y = self.conv3(y)
        y = self.norm3(y)

        if self.in_channels != self.out_channels:
            residual = self.skip_conv(residual)
        y += residual
        y = self.nl_layer(y)
        return y
