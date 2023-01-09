from module.baseline.base_hrnet.hrnet_encoder import HRNetEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from ever.core import registry
import ever as er
from module.CGFL import SegmentationLoss

BatchNorm2d = nn.BatchNorm2d

BN_MOMENTUM = 0.1


class SimpleFusion(nn.Module):
    def __init__(self, in_channels):
        super(SimpleFusion, self).__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

    def forward(self, feat_list):
        # print(feat_list[0].shape)
        # print(feat_list[1].shape)
        # print(feat_list[2].shape)
        # print(feat_list[3].shape)
        # torch.Size([16, 32, 128, 128])
        # torch.Size([16, 64, 64, 64])
        # torch.Size([16, 128, 32, 32])
        # torch.Size([16, 256, 16, 16])

        x0 = feat_list[0]
        x0_h, x0_w = x0.size(2), x0.size(3)
        x1 = F.interpolate(feat_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(feat_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(feat_list[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = self.fuse_conv(x)
        return x


@registry.MODEL.register('rs3NetFusion')
class rsNetFusion(er.ERModule):
    def __init__(self, config):
        super(rsNetFusion, self).__init__(config)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.backbone = HRNetEncoder(self.config.backbone)
        self.neck = ASFF9(self.config.neck.in_channels)
        self.head = nn.Sequential(
            nn.Conv2d(self.config.head.in_channels, self.config.classes, 1),
            nn.UpsamplingBilinear2d(scale_factor=self.config.head.upsample_scale),
        )
        self.loss = SegmentationLoss(self.config.loss)

    def forward(self, x, y=None):

        pred_list = self.backbone(x)


        logit = self.neck(pred_list)
        # print('logit',logit.shape)    torch.Size([16, 480, 128, 128])
        #logit = self.head(logit)
        # print('logit', logit.shape) torch.Size([16, 7, 512, 512])

        if self.training:
            y_true = y['cls']
            return self.loss(logit, y_true.long())
        else:
            return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                hrnet_type='hrnetv2_w48',
                pretrained=False,
                norm_eval=False,
                frozen_stages=-1,
                with_cp=False,
                with_gc=False,
            ),
            neck=dict(
                in_channels=720,
            ),
            classes=7,
            head=dict(
                in_channels=720,
                upsample_scale=4.0,
            ),
            loss=dict(

                ce=dict(),
            )
        ))



def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class ASFF(nn.Module):
    def __init__(self, vis=False):
        super(ASFF, self).__init__()

        self.dim = [256, 128, 64]
        #self.expand = add_conv(480, 128, 3, 1)

        compress_c = 16
            #8 if rfb else 16  #when adding rfb, we use half number of channels to save memory


        self.weight_levels = nn.Conv2d(compress_c*4, 4, kernel_size=1, stride=1, padding=0)
        self.vis= vis

        self.weight_level_0 = nn.Conv2d(in_channels=256,out_channels=compress_c, kernel_size=1, stride=1,padding=0, bias=False)
        self.atrous_block1 = nn.Conv2d(32, compress_c, 1, 1)
        self.atrous_block6 = nn.Conv2d(64, compress_c, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(128, compress_c, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(256, compress_c, 3, 1, padding=18, dilation=18)

    def forward(self, feat_list):
        # print(x_level_0.shape)
        # print(x_level_1.shape)
        # print(x_level_2.shape)
        # print(x_level_3.shape)
        # torch.Size([16, 32, 128, 128])
        # torch.Size([16, 64, 64, 64])
        # torch.Size([16, 128, 32, 32])
        # torch.Size([16, 256, 16, 16])
        x_level_0 = feat_list[0]
        x_level_1 = feat_list[1]
        x_level_2 = feat_list[2]
        x_level_3 = feat_list[3]
        x_level_3 = F.interpolate(x_level_3, scale_factor=8, mode='nearest')
        x_level_2 = F.interpolate(x_level_2, scale_factor=4, mode='nearest')
        x_level_1 = F.interpolate(x_level_1, scale_factor=2, mode='nearest')

        level_0_weight_v = self.atrous_block1(x_level_0)
        level_1_weight_v = self.atrous_block6(x_level_1)
        level_2_weight_v = self.atrous_block12(x_level_2)
        level_3_weight_v = self.atrous_block18(x_level_3)
        # print(level_0_weight_v.shape)
        # print(level_1_weight_v.shape)
        # print(level_2_weight_v.shape)
        # print(level_3_weight_v.shape)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v,level_3_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        #print(x_level_0.shape)
        #print(x_level_1.shape)
        #print(x_level_2.shape)
        # ca0 = self.ca0(x_level_0)
        # ca1 = self.ca1(x_level_1)
        # ca2 = self.ca2(x_level_2)
        # ca3 = self.ca3(x_level_3)
        # fused_out_reduced = torch.cat((x_level_0 *ca0* levels_weight[:,0:1,:,:], x_level_1 * ca1*levels_weight[:,1:2,:,:], x_level_2 * ca2*levels_weight[:,2:3,:,:],x_level_3 * ca3*levels_weight[:,3:,:,:]),1)
        fused_out_reduced = torch.cat((x_level_0 * levels_weight[:,0:1,:,:], x_level_1 *levels_weight[:,1:2,:,:], x_level_2 *levels_weight[:,2:3,:,:],x_level_3 *levels_weight[:,3:,:,:]),1)
        #print(fused_out_reduced.shape)
        #out = self.expand(fused_out_reduced)
        #print(fused_out_reduced.shape)
        return fused_out_reduced
class ASFF2(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF2, self).__init__()
        self.level = level
        self.dim = [256, 128, 64]
        self.inter_dim = self.dim[self.level]


        self.stride_level_1 = add_conv(128, self.inter_dim, 1, 1)
        self.compress_level_0 = add_conv(256, self.inter_dim, 1, 1)
        self.expand = add_conv(1024, 128, 3, 1)

        compress_c = 16
            #8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(64, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*4, 4, kernel_size=1, stride=1, padding=0)
        self.vis= vis


        #self.weight_level_0 = add_conv(256, compress_c, 1, 1)
        self.weight_level_0 = nn.Conv2d(in_channels=256,out_channels=compress_c, kernel_size=1, stride=1,padding=0, bias=False)


        self.atrous_block1 = nn.Conv2d(256, compress_c, 1, 1)
        self.atrous_block6 = nn.Conv2d(256, compress_c, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(256, compress_c, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(256, compress_c, 3, 1, padding=18, dilation=18)

        self.mlp = Mlp(in_features=1024, hidden_features=256, drop=0.)

        self.end = nn.Conv2d(256, 128, kernel_size=1)
    def forward(self, x_level_3, x_level_2, x_level_1,x_level_0):
        # print(x_level_0.shape)
        # print(x_level_1.shape)
        # print(x_level_2.shape)
        # print(x_level_3.shape)
        # torch.Size([2, 256, 224, 224])
        # torch.Size([2, 256, 112, 112])
        # torch.Size([2, 256, 56, 56])
        # torch.Size([2, 256, 28, 28])

        x_level_0 = F.interpolate(x_level_0, scale_factor=8, mode='nearest')
        x_level_1 = F.interpolate(x_level_1, scale_factor=4, mode='nearest')
        x_level_2 = F.interpolate(x_level_2, scale_factor=2, mode='nearest')

        level_0_weight_v = self.atrous_block1(x_level_0)
        level_1_weight_v = self.atrous_block6(x_level_1)
        level_2_weight_v = self.atrous_block12(x_level_2)
        level_3_weight_v = self.atrous_block18(x_level_3)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v,level_3_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        #print(x_level_0.shape)
        #print(x_level_1.shape)
        #print(x_level_2.shape)
        # ca0 = self.ca0(x_level_0)
        # ca1 = self.ca1(x_level_1)
        # ca2 = self.ca2(x_level_2)
        # ca3 = self.ca3(x_level_3)
        # fused_out_reduced = torch.cat((x_level_0 *ca0* levels_weight[:,0:1,:,:], x_level_1 * ca1*levels_weight[:,1:2,:,:], x_level_2 * ca2*levels_weight[:,2:3,:,:],x_level_3 * ca3*levels_weight[:,3:,:,:]),1)
        fused_out_reduced = torch.cat((x_level_0 * levels_weight[:,0:1,:,:], x_level_1 *levels_weight[:,1:2,:,:], x_level_2 *levels_weight[:,2:3,:,:],x_level_3 *levels_weight[:,3:,:,:]),1)
        #print(fused_out_reduced.shape)([2, 1024, 224, 224])
        #out = self.expand(fused_out_reduced)
        B, C, H, W = fused_out_reduced.shape
        fused_out_reduced = fused_out_reduced.view(2, H * W, C)
        #print(fused_out_reduced.shape)
        out = self.mlp(fused_out_reduced)
        #print(out.shape)[2, 50176, 128]
        x = out.permute(0, 2, 1)
        x = x.contiguous().view(B, 256, H, W)
        #print(x.shape)2, 128, 224, 224
        x = self.end(x)
        return x
class ASFF3(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF3, self).__init__()
        self.level = level
        self.dim = [256, 128, 64]
        self.inter_dim = self.dim[self.level]


        self.stride_level_1 = add_conv(128, self.inter_dim, 1, 1)
        self.compress_level_0 = add_conv(256, self.inter_dim, 1, 1)
        self.expand = add_conv(256, 128, 3, 1)

        compress_c = 16
            #8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(64, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*4, 4, kernel_size=1, stride=1, padding=0)
        self.vis= vis


        #self.weight_level_0 = add_conv(256, compress_c, 1, 1)
        self.weight_level_0 = nn.Conv2d(in_channels=256,out_channels=compress_c, kernel_size=1, stride=1,padding=0, bias=False)


        self.atrous_block1 = nn.Conv2d(256, compress_c, 1, 1)
        self.atrous_block6 = nn.Conv2d(256, compress_c, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(256, compress_c, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(256, compress_c, 3, 1, padding=18, dilation=18)

        # self.convT = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        # self.convT2 = nn.ConvTranspose2d(64, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        relu = nn.PReLU(num_parameters=1, init=0.25)
        conv = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        bn = nn.GroupNorm(64, 64)
        up = [conv, bn, relu]
        self.up = nn.Sequential(*up)
        conv2 = nn.ConvTranspose2d(64, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        bn2 = nn.GroupNorm(16, 16)
        up2 = [conv2]
        self.up2 = nn.Sequential(*up2)

    def forward(self, x_level_3, x_level_2, x_level_1,x_level_0):
        # print(x_level_0.shape)
        # print(x_level_1.shape)
        # print(x_level_2.shape)
        # print(x_level_3.shape)
        # torch.Size([2, 256, 224, 224])
        # torch.Size([2, 256, 112, 112])
        # torch.Size([2, 256, 56, 56])
        # torch.Size([2, 256, 28, 28])

        x_level_0 = F.interpolate(x_level_0, scale_factor=8, mode='nearest')
        x_level_1 = F.interpolate(x_level_1, scale_factor=4, mode='nearest')
        x_level_2 = F.interpolate(x_level_2, scale_factor=2, mode='nearest')

        level_0_weight_v = self.atrous_block1(x_level_0)
        level_1_weight_v = self.atrous_block6(x_level_1)
        level_2_weight_v = self.atrous_block12(x_level_2)
        level_3_weight_v = self.atrous_block18(x_level_3)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v,level_3_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        # fused_out_reduced = torch.cat((x_level_0 * levels_weight[:,0:1,:,:], x_level_1 *levels_weight[:,1:2,:,:], x_level_2 *levels_weight[:,2:3,:,:],x_level_3 *levels_weight[:,3:,:,:]),1)
        fused_out_reduced = x_level_0 * levels_weight[:,0:1,:,:]+ x_level_1 *levels_weight[:,1:2,:,:]+ x_level_2 *levels_weight[:,2:3,:,:]+x_level_3 *levels_weight[:,3:,:,:]
        #print(fused_out_reduced.shape)
        out = self.expand(fused_out_reduced)
        out = self.up(out)
        out = self.up2(out)
        #print(out.shape)
        return out


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim


        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

class ASFF9(nn.Module):
    def __init__(self, vis=False):
        super(ASFF9, self).__init__()

        self.dim = [256, 128, 64]
        #self.expand = add_conv(480, 128, 3, 1)

        compress_c = 8
            #8 if rfb else 16  #when adding rfb, we use half number of channels to save memory


        self.weight_levels = nn.Conv2d(compress_c*4, 4, kernel_size=1, stride=1, padding=0)
        self.vis= vis

        self.weight_level_0 = nn.Conv2d(in_channels=256,out_channels=compress_c, kernel_size=1, stride=1,padding=0, bias=False)
        self.atrous_block1 = nn.Conv2d(32, compress_c, 1, 1)
        self.atrous_block6 = nn.Conv2d(64, compress_c, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(128, compress_c, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(256, compress_c, 3, 1, padding=18, dilation=18)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(480, 480, 1),
            nn.BatchNorm2d(480),
            nn.ReLU(True)
        )
        #self.attn = Self_Attn(128)

        # self.head = nn.Sequential(
        #     nn.Conv2d(self.config.head.in_channels, self.config.classes, 1),
        #     nn.UpsamplingBilinear2d(scale_factor=self.config.head.upsample_scale),
        # )
        relu = nn.PReLU(num_parameters=1, init=0.25)
        conv = nn.ConvTranspose2d(128, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        bn = nn.GroupNorm(64, 64)
        up = [conv]
        self.up = nn.Sequential(*up)
        conv2 = nn.ConvTranspose2d(32, 7, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        up2 = [conv2]
        self.up2 = nn.Sequential(*up2)
        self.expand = add_conv(480, 128, 3, 1)
    def forward(self, feat_list):
        # print(x_level_0.shape)
        # print(x_level_1.shape)
        # print(x_level_2.shape)
        # print(x_level_3.shape)
        # torch.Size([16, 32, 128, 128])
        # torch.Size([16, 64, 64, 64])
        # torch.Size([16, 128, 32, 32])
        # torch.Size([16, 256, 16, 16])
        x_level_0 = feat_list[0]
        x_level_1 = feat_list[1]
        x_level_2 = feat_list[2]
        x_level_3 = feat_list[3]
        x_level_3 = F.interpolate(x_level_3, scale_factor=8, mode='nearest')
        x_level_2 = F.interpolate(x_level_2, scale_factor=4, mode='nearest')
        x_level_1 = F.interpolate(x_level_1, scale_factor=2, mode='nearest')

        level_0_weight_v = self.atrous_block1(x_level_0)
        level_1_weight_v = self.atrous_block6(x_level_1)
        level_2_weight_v = self.atrous_block12(x_level_2)
        level_3_weight_v = self.atrous_block18(x_level_3)
        # print(level_0_weight_v.shape)
        # print(level_1_weight_v.shape)
        # print(level_2_weight_v.shape)
        # print(level_3_weight_v.shape)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v,level_3_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        #print(x_level_0.shape)
        #print(x_level_1.shape)
        #print(x_level_2.shape)
        # ca0 = self.ca0(x_level_0)
        # ca1 = self.ca1(x_level_1)
        # ca2 = self.ca2(x_level_2)
        # ca3 = self.ca3(x_level_3)
        # fused_out_reduced = torch.cat((x_level_0 *ca0* levels_weight[:,0:1,:,:], x_level_1 * ca1*levels_weight[:,1:2,:,:], x_level_2 * ca2*levels_weight[:,2:3,:,:],x_level_3 * ca3*levels_weight[:,3:,:,:]),1)
        ca0 = x_level_0 * levels_weight[:,0:1,:,:]
        ca1 = x_level_1 *levels_weight[:,1:2,:,:]
        ca2 = x_level_2 *levels_weight[:,2:3,:,:]
        ca3 = x_level_3 *levels_weight[:,3:,:,:]
        # print(ca0.shape)
        # print(ca1.shape)
        # print(ca2.shape)
        # print(ca3.shape)
        # torch.Size([8, 32, 128, 128])
        # torch.Size([8, 64, 128, 128])
        # torch.Size([8, 128, 128, 128])
        # torch.Size([8, 256, 128, 128])
        fused_out_reduced = torch.cat((ca0, ca1, ca2, ca3),1)

        out = self.expand(fused_out_reduced)
        out = self.up(out)
        out = self.up2(out)




        # fused_out_reduced = torch.cat((x_level_0 * levels_weight[:,0:1,:,:], x_level_1 *levels_weight[:,1:2,:,:], x_level_2 *levels_weight[:,2:3,:,:],x_level_3 *levels_weight[:,3:,:,:]),1)
        #print(fused_out_reduced.shape)
        #out = self.fuse_conv(fused_out_reduced)
        #print(fused_out_reduced.shape)
        return out