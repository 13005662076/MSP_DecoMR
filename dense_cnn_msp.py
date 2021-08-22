import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np
from mindspore import ops

from resnet_msp import resnet50backbone
from layer_msp import ConvBottleNeck
from ResizeBilinear import *
import os

        


def warp_feature(dp_out,feature_map,uv_res):
    assert dp_out.shape[0] == feature_map.shape[0]
    assert dp_out.shape[2] == feature_map.shape[2]
    assert dp_out.shape[3] == feature_map.shape[3]
    
    expand_dims=ops.ExpandDims()
    dp_mask =expand_dims(dp_out[:,0],1)
    dp_uv=dp_out[:,1:]
    B,C,H,W =feature_map.shape
    device =feature_map.device

    index_batch=mindspore.numpy.arange(0,B,device=device,dtype=mindspore.int64)
    index_batch=index_batch.view(-1).astype("int64")

    tmp_x = mindspore.numpy.arange(0,W,device=device,dtype=mindspore.int64)
    tmp_y = mindspore.numpy.arange(0,H,device=device,dtype=mindspore.int64)

    meshgrid = ops.Meshgrid(indexing="ij")
    y,x = meshgrid((tmp_y, tmp_x))
    y = y.view(-1).repeat([B])
    x = x.view(-1).repeat([B])

    conf = dp_mask[index_batch, 0, y, x]
    valid=conf>thre
    index_batch =  index_batch[valid]
    x = x[valid]
    y = y[valid]

    uv=dp_uv[index_batch, :, y, x]
    num_pixel = uv.shape[0]

    uv = uv*(uv_res-1)
    m_round=ops.Round()
    uv_round = m_round(uv).astype("int64").clip(xmin=0, xmax=uv_res - 1)

    index_uv = (uv_round[:, 1] * uv_res + uv_round[:, 0]).copy()+ index_batch * uv_res * uv_res

    sampled_feature = feature_map[index_batch, :, y, x]

    y = (2 * y.astype("float32") / (H - 1)) - 1
    x = (2 * x..astype("float32") / (W - 1)) - 1
    concat=ops.Concat(-1) #dim=1
    sampled_feature = concat([sampled_feature, x[:, None], y[:, None]], dim=-1)

    
    dp_mask=dp_out
    return

#DPNet:returns densepose result
class DPNet(nn.Cell):
    def __init__(self,warp_lv=2,norm_type="BN"):
        super(DPNet,self).__init__()
        nl_layer=nn.ReLU()
        self.warp_lv=warp_lv

        #image encoder
        self.resnet=resnet50backbone(pretrained=False)

        dp_layers = []
        
        channel_list = [3, 64, 256, 512, 1024, 2048]
        for i in range(warp_lv,5):
            in_channels=channel_list[i+1]
            out_channels=channel_list[i]

            dp_layers.append(
                nn.SequentialCell(
                    #nn.ResizeBilinear(),
                    ResizeBilinear(scale_factor=2),
                    ConvBottleNeck(in_channels=in_channels,out_channels=out_channels,nl_layer=nl_layer, norm_type=norm_type)
                    )
                )
        self.dp_layers=nn.CellList(dp_layers)
        
        self.dp_uv_end=nn.SequentialCell(ConvBottleNeck(channel_list[warp_lv], 32, nl_layer,  norm_type=norm_type),
                                         nn.Conv2d(32,2,kernel_size=1),
                                         nn.Sigmoid()
                                         )
        self.dp_mask_end = nn.SequentialCell(ConvBottleNeck(channel_list[warp_lv], 32, nl_layer, norm_type=norm_type),
                                         nn.Conv2d(32, 1, kernel_size=1),
                                         nn.Sigmoid()
                                         )
    def construct(self,image,UV=None):
        codes,features=self.resnet(image)
        dp_feature=features[-1]
        for i in range(len(self.dp_layers)-1,-1,-1):
            
            
            '''
            if isinstance(self.dp_layers[i][0],nn.ResizeBilinear):
                self.dp_layers[i][0](scale_factor=2)
            '''

            dp_feature=self.dp_layers[i](dp_feature)
            dp_feature=dp_feature + features[i - 1 + len(features) - len(self.dp_layers)]
        dp_uv=self.dp_uv_end(dp_feature)
        dp_mask=self.dp_mask_end(dp_feature)
        ops_cat=ops.Concat(1)
        dp_out=ops_cat((dp_mask,dp_uv))
        return dp_out,dp_feature,codes

def get_LNet(options):
    if options.model == "DecoMR":
        uv_net = UVNet(uv_channels=options.uv_channels,
                       uv_res=options.uv_res,
                       warp_lv=options.warp_level,
                       uv_type=options.uv_type,
                       norm_type=options.norm_type
                       )

# UVNet returns location map
class UVNet(nn.Cell):
    def __init__(self,uv_channels=64,uv_res=128,warp_lv=2,uv_type="SMPL",norm_type="BN"):
        super(UVNet,self).__init__()

        nl_layer = nn.ReLU()
        
        self.fc_head = nn.SequentialCell(
            nn.Dense(2048,512),
            nn.BatchNorm1d(512),
            nl_layer,
            nn.Dense(512,256)
            )
        self.camera = nn.SequentialCell(
            nn.Dense(2048,512),
            nn.BatchNorm1d(512),
            nl_layer,
            nn.Dense(512,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dense(256,3)
            )

        self.warp_lv =warp_lv
        channel_list = [3, 64, 256, 512, 1024, 2048]
        warp_channel = channel_list[warp_lv]
        self.uv_res = uv_res
        self.warp_res = int(256 // (2**self.warp_lv))

        if uv_type == "SMPL":
            ref_file = 'data/SMPL_ref_map_{0}.npy'.format(self.warp_res)
        elif uv_type == 'BF':
            ref_file = 'data/BF_ref_map_{0}.npy'.format(self.warp_res)

        if not os.path.exists(ref_file):
            sampler = Index_UV_Generator(UV_height=self.warp_res, uv_type=uv_type)
            ref_vert, _ = read_obj('data/reference_mesh.obj')
            ref_map = sampler.get_UV_map(torch.FloatTensor(ref_vert))
            np.save(ref_file, ref_map.cpu().asnumpy())
        self.ref_map = Tensor(np.load(ref_file),dtype=mindspore.float32).transpose(0, 3, 1, 2)

        self.uv_conv1 = nn.SequentialCell(
            nn.Conv2d(256 + warp_channel + 3 + 3, 2 * warp_channel, kernel_size=1),
            nl_layer,
            nn.Conv2d(2 * warp_channel, 2 * warp_channel, kernel_size=1),
            nl_layer,
            nn.Conv2d(2 * warp_channel, warp_channel, kernel_size=1)
            )

        uv_lv = 0 if uv_res == 256 else 1
        self.hg = HgNet(in_channels=warp_channel, level=5 - warp_lv, nl_layer=nl_layer, norm_type=norm_type)

        cur = min(8, 2 ** (warp_lv - uv_lv))
        prev = cur
        self.uv_conv2 = ConvBottleNeck(warp_channel, uv_channels * cur, nl_layer, norm_type=norm_type)

        layers = []
        for lv in range(warp_lv, uv_lv, -1):
            cur = min(prev, 2 ** (lv - uv_lv - 1))
            layers.append(
                nn.SequentialCell(ResizeBilinear(scale_factor=2),
                              ConvBottleNeck(uv_channels * prev, uv_channels * cur, nl_layer, norm_type=norm_type)
                                  )
            )
            prev = cur
        self.decoder = nn.Sequential(*layers)
        self.uv_end = nn.Sequential(ConvBottleNeck(uv_channels, 32, nl_layer, norm_type=norm_type),
                                    nn.Conv2d(32, 3, kernel_size=1)
                                    )
    def construct(self,dp_out,dp_feature,codes):
        n_batch = dp_out.shape[0]
        local_feature = warp_feature(dp_out, dp_feature, self.warp_res)
        
        global_feature = self.fc_head(codes)
        
        global_feature = global_feature[:, :, None, None].expand(-1, -1, self.warp_res, self.warp_res)
        self.ref_map = self.ref_map.astype(local_feature.dtype)

        concat=ops.Concat(1)
        uv_map = concat([local_feature, global_feature, self.ref_map.expand(n_batch, -1, -1, -1)])
        uv_map = self.uv_conv1(uv_map)
        uv_map = self.hg(uv_map)
        uv_map = self.uv_conv2(uv_map)
        uv_map = self.decoder(uv_map)
        uv_map = self.uv_end(uv_map).transpose(0, 2, 3, 1)

        cam = self.camera(codes)
        return uv_map, cam



import cv2
img=cv2.imread("src.jpg")
img=cv2.resize(img,(224,224))
img=mindspore.Tensor(img).astype("float32")
expand_dims=ops.ExpandDims()
img=expand_dims(img.transpose(2,0,1),0)#.expand(30,-1,-1,-1)
model=DPNet()
dp_out, dp_feature, codes=model(img)

print(type(dp_out))

print(dp_out.asnumpy())
print(dp_feature.shape)
print(codes.shape)

'''
lmap=warp_feature(dp_out, dp_feature,int(256 // (2 ** 2)))
print(lmap.size())
img=lmap[0].permute(2,0,1).detach().numpy()
print(img.shape)
cv2.imshow("",img)
'''
