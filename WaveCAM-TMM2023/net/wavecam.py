

import os
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_


import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
import torch.nn.functional as F


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'wave_T': _cfg(crop_pct=0.9),
    'wave_S': _cfg(crop_pct=0.9),
    'wave_M': _cfg(crop_pct=0.9),
    'wave_B': _cfg(crop_pct=0.875),
}






class WaveModeling(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mode='fc'):
        super().__init__()

        self.fc_h = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_c = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.tfc_h = nn.Conv2d(2 * dim, dim, (1, 7), stride=1, padding=(0, 7 // 2), groups=dim, bias=False)
        self.tfc_w = nn.Conv2d(2 * dim, dim, (7, 1), stride=1, padding=(7 // 2, 0), groups=dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.w1 = nn.Conv2d(20, 20, 1, 1)
        self.w2 = nn.Conv2d(20, 20, 1, 1)
        if mode == 'fc':
            self.theta_R_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
            self.theta_I_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())

    def forward(self, x):

        B, C, H, W = x.shape
        x = F.relu(x / 3 + 0.1)

        theta_R = self.theta_R_conv(x)

        background =  (1 - x) / 3
        theta_I = self.theta_I_conv(background)

        x_h = self.fc_h(x)
        x_w = self.fc_w(background)
        x_h = torch.cat([x_h * torch.cos(theta_R), x_h * torch.sin(theta_R)], dim=1)
        x_w = torch.cat([x_w * torch.cos(theta_I), x_w * torch.sin(theta_I)], dim=1)

        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)

        a = F.adaptive_avg_pool2d(x , output_size=1)

        w1 = self.w1(a)
        w2 = self.w2(a)
        a = torch.cat([ w1,  w2], dim=1)
        a = a.reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)

        x = torch.cat([h * a[0], w * a[1]],dim=1)
        x = self.proj_drop(x)
        return x
