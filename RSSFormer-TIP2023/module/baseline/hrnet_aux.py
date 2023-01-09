from module.baseline.base_hrnet.hrnet_encoder import HRNetEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from ever.core import registry
import ever as er
from module.CGFL import SegmentationLossaux as SegmentationLoss

# from module.newloss2 import SegmentationLoss
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
class SimpleFusion8(nn.Module):
    def __init__(self, in_channels):
        super(SimpleFusion8, self).__init__()
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
        return x,x0

@registry.MODEL.register('RSSFormer')
class HRNetFusion(er.ERModule):
    def __init__(self, config):
        super(HRNetFusion, self).__init__(config)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.backbone = HRNetEncoder(self.config.backbone)
        #self.neck = SimpleFusion(self.config.neck.in_channels)
        self.neck = SimpleFusion8(self.config.neck.in_channels)
        self.head = nn.Sequential(
            nn.Conv2d(self.config.head.in_channels, self.config.classes, 1),
            nn.UpsamplingBilinear2d(scale_factor=self.config.head.upsample_scale),
        )
        self.loss = SegmentationLoss(self.config.loss)

        # self.headaux = nn.Sequential(nn.Linear(480, 128),
        #                           nn.Linear(128, 7))
        self.headaux = nn.Sequential( nn.Linear(32, 7))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, y=None):
        pred_list = self.backbone(x)
        # print('the flops is G=====================================')
        # from thop import profile
        # input = torch.randn(1, 3, 224, 224).cuda()
        # flops, params = profile(self.backbone, inputs=(input,))
        # print('the flops is {}G,the params is {}M'.format(round(flops / (10 ** 9), 2), round(params / (10 ** 6), 2)))
        #logit = self.neck(pred_list)
        logit,f0 = self.neck(pred_list)
        # print('logit',logit.shape)    #torch.Size([16, 480, 128, 128])
        x= self.avg_pool(f0)
        #print('x', x.flatten(1).shape)
        logits = self.headaux(x.flatten(1))

        logit = self.head(logit)
        # print('logit', logit.shape) torch.Size([16, 7, 512, 512])

        if self.training:
            y_true = y['cls']
            return self.loss(logit, y_true.long(),logits)#.softmax(dim=1))
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


