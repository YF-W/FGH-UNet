import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import torch.fft
from torchvision import models as resnet_model
from thop import profile

"""
FGH-UNet

Proposed By Yuefei Wang, Li Zhang, Yutong Zhang
Chengdu University
2024. 3

"""
"""For research and clinical study only, commercial use is strictly prohibited"""

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv_empty(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv_empty, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, dilation=2,padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3,dilation=2,padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out



def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)




class GlobalLocalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear',
                                   align_corners=True).permute(1, 2, 3, 0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


class BottleNeck(nn.Module):
    def __init__(self):
        super(BottleNeck,self).__init__()
        self.conv1_1 =DepthwiseSeparableConv(1024,512)
        self.conv1_2 =DepthwiseSeparableConv(512,256)

        self.conv2_1 = DoubleConv(1024,512)
        self.conv2_2 = DoubleConv(512,256)

        self.conv3_1 = DoubleConv_empty(1024,512)
        self.conv3_2 = DoubleConv_empty(512,256)


        self.cha_atten = ChannelGate(512)
        self.spa_atten = SpatialGate()

        self.conv1x1 = nn.Conv2d(768,512,1)
        self.adjust_conv = DoubleConv(256,512)


    def forward(self,x):



        x1 = x2 = x3 = x


        x1_cha_atten = self.cha_atten(x1)
        x1_spa_attem = self.spa_atten(x1)
        x1_1 = x1_cha_atten * x1
        x1_2 = x1_spa_attem * x1
        x1 = torch.cat([x1_1,x1_2],dim=1)
        x1 = self.conv1_1(x1)
        x1 = self.conv1_2(x1)


        x2_cha_atten = self.cha_atten(x2)
        x2_spa_attem = self.spa_atten(x2)
        x2_1 = x2_cha_atten * x2
        x2_2 = x2_spa_attem * x2
        x2 = torch.cat([x2_1,x2_2],dim=1)
        x2 = self.conv2_1(x2)
        x2 = self.conv2_2(x2)

        x3_cha_atten = self.cha_atten(x3)
        x3_spa_attem = self.spa_atten(x3)
        x3_1 = x3_cha_atten * x3
        x3_2 = x3_spa_attem * x3
        x3 = torch.cat([x3_1,x3_2],dim=1)
        x3 = self.conv3_1(x3)
        x3 = self.conv3_2(x3)

        x = x1+x2+x3
        x = self.adjust_conv(x)



        return x


class muti_conv (nn.Module):
    def __init__(self,in_chas,out_chas):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chas,in_chas//4,1)
        self.conv2 = nn.Conv2d(in_chas//4,in_chas,1)
        self.sigmoid = nn.Sigmoid()

        self.dconv2 = DoubleConv(in_chas//4,in_chas//4)
        self.upconv2 = nn.ConvTranspose2d(in_chas,in_chas//4,2,2)
        self.conv3  = nn.Conv2d(in_chas,in_chas,3,2,1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dconv1 = DoubleConv(in_chas, in_chas * 4)
        self.upconv1 = nn.ConvTranspose2d(in_chas * 4, out_chas, 2, 2)

    def forward(self,x):

        x1 = x2 = x


        x1_1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)
        x1_1 = self.conv1(x1_1)

        x1_2 = self.upconv2(x1)
        x1_2 = self.dconv2(x1_2)

        x1 = x1_1 + x1_2
        x1 = self.pool(x1)
        x1 = self.conv2(x1)
        x1 = self.sigmoid(x1)


        x2 = self.pool(x2)
        x2 = self.dconv1(x2)
        x2 = self.upconv1(x2)

        x = torch.cat([x,x1],dim=1)
        x = x+x2

        return x



class gnconv(nn.Module):
    def __init__(self, dim, order, gflayer, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [muti_conv(self.dims[i], self.dims[i + 1]) for i in range(order - 1)]
        )

        self.scale = s
        print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x



class Block(nn.Module):
    r""" HorNet block
    """

    def __init__(self, dim,order,gflayer,drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim,order,gflayer)  # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x




class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x




class baseline(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = resnet_model.resnet34(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        # self.pool = resnet.maxpool

        self.e_layer1 = resnet.layer1
        self.e_layer2 = resnet.layer2
        self.e_layer3 = resnet.layer3
        self.e_layer4 = resnet.layer4

        self.hor_layer1 = Block(dim=64, order=2, gflayer=None, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv)
        self.hor_layer2 = Block(dim=128, order=3, gflayer=None, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv)
        self.hor_layer3 = Block(dim=256, order=4, gflayer=None, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv)
        self.hor_layer4 = Block(dim=512, order=5, gflayer=None, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv)

        self.pool1 = nn.Conv2d(64,128,3,2,1)
        self.pool2 = nn.Conv2d(128,256,3,2,1)
        self.pool3 = nn.Conv2d(256,512,3,2,1)
        self.avgpool = nn.AvgPool2d(2,2)

        self.bottleneck =BottleNeck()
        self.up1 = nn.ConvTranspose2d(1024,512,2,2)
        self.up2 = nn.ConvTranspose2d(512,256,2,2)
        self.up3 = nn.ConvTranspose2d(256,128,2,2)
        self.up4 = nn.ConvTranspose2d(128,64,2,2)
        self.up5 = nn.ConvTranspose2d(64,32,2,2)

        self.dconv1 = DoubleConv(1024,512)
        self.dconv2 = DoubleConv(512,256)
        self.dconv3 = DoubleConv(256,128)
        self.dconv4 = DoubleConv(128,64)
        self.dconv5 = DoubleConv(32,32)

        self.out_conv = nn.Conv2d(32,1,3,1,1)


    def forward(self,x):

        skip_original = []


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.pool(x)


        e_layer1 = self.e_layer1(x)
        skip_original.append(e_layer1)
        hor_layer1_1 = self.hor_layer1(e_layer1) #此输出(与e_layer1相同)要和e_layer1做处理作为跳连接,以下类似[4,64,112,112]
        hor_layer1_2 = self.pool1(hor_layer1_1)  #经过调整尺寸大小和通道数，最为horlayer2的输入[4,128,56,56]

        e_layer2 = self.e_layer2(e_layer1)
        skip_original.append(e_layer2)
        hor_layer2_1 = self.hor_layer2(hor_layer1_2)
        hor_layer2_2 = self.pool2(hor_layer2_1)
        e_layer2 = e_layer2


        e_layer3 = self.e_layer3(e_layer2)
        skip_original.append(e_layer3)
        hor_layer3_1 = self.hor_layer3(hor_layer2_2)
        hor_layer3_2 = self.pool3(hor_layer3_1)
        e_layer3 = e_layer3

        e_layer4 = self.e_layer4(e_layer3)
        skip_original.append(e_layer4)
        hor_layer4 = self.hor_layer4(hor_layer3_2)
        e_end = e_layer4

        # bottleneck = self.avgpool(e_end)
        bottleneck = self.bottleneck(e_end)


        d_layer1 = bottleneck
        d_layer1 = torch.cat([d_layer1,(hor_layer4+skip_original[3])],dim=1)
        d_layer1 = self.dconv1(d_layer1)

        d_layer2 = self.up2(d_layer1)
        d_layer2 = torch.cat([d_layer2,(hor_layer3_1+skip_original[2])],dim=1)
        d_layer2 = self.dconv2(d_layer2)

        d_layer3 = self.up3(d_layer2)
        d_layer3 = torch.cat([d_layer3, (hor_layer2_1 + skip_original[1])], dim=1)
        d_layer3 = self.dconv3(d_layer3)

        d_layer4 = self.up4(d_layer3)
        d_layer4 = torch.cat([d_layer4, (hor_layer1_1 + skip_original[0])], dim=1)
        d_layer4 = self.dconv4(d_layer4)

        d_layer5 = self.up5(d_layer4)
        d_layer5 = self.dconv5(d_layer5)

        x = self.out_conv(d_layer5)

        return x




x = torch.randn([4,3,224,224])
model = baseline()
# print(model(x).shape)
flops, params = profile(model, inputs=(x,))
print(f'Flops: {flops}, params: {params}')



