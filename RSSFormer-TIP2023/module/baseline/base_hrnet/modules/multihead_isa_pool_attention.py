import os
import pdb
import math
import torch
import torch.nn as nn
#from .multihead_attention import MultiheadAttention as MHA_
from .DAL import Mhca as MHA_
import torch.nn.functional as F
from .multihead_isa_attention import PadBlock, LocalPermuteModule

class InterlacedPoolAttention(nn.Module):
    r""" interlaced sparse multi-head self attention (ISA) module with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): Window size.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self, embed_dim, num_heads, window_size=7,
                 rpe=True, **kwargs):
        super(InterlacedPoolAttention, self).__init__()
        
        self.dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.with_rpe = rpe


        self.attn = MHA_(embed_dim, num_heads, **kwargs)
        self.pad_helper = PadBlock(window_size)
        self.permute_helper = LocalPermuteModule(window_size)

    def forward(self, x, H, W, **kwargs):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        # attention
        # pad
        x_pad = self.pad_helper.pad_if_needed(x, x.size())
        # permute
        x_permute = self.permute_helper.permute(x_pad, x_pad.size())
        # attention
        out = self.attn(x_permute, x_permute, x_permute, **kwargs)
        # reverse permutation
        out = self.permute_helper.rev_permute(out, x_pad.size())
        out = self.pad_helper.depad_if_needed(out, x.size())
        return out.reshape(B, N, C)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),

            nn.PReLU(in_size//reduction),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            nn.PReLU(in_size)
        )

    def forward(self, x):
        return x * self.se(x)


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert  kernel_size in (3,7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class InterlacedPoolAttention2(nn.Module):
    r""" interlaced sparse multi-head self attention (ISA) module with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): Window size.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, embed_dim, num_heads, window_size=7,
                 rpe=True, **kwargs):
        super(InterlacedPoolAttention2, self).__init__()

        self.dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.with_rpe = rpe
        #print('====================================',embed_dim)
        self.attn = MHA_(embed_dim, num_heads, **kwargs)
        self.pad_helper = PadBlock(window_size)
        self.permute_helper = LocalPermuteModule(window_size)


        self.atrous_block1 = SpatialAttention(7) #nn.Conv2d(self.dim, 8, 1, 1)
        self.atrous_block2 = SpatialAttention(7)   #nn.Conv2d(self.dim, 8, 1, 1)
        self.weight_levels = nn.Conv2d(1 * 2, 2, kernel_size=1, stride=1, padding=0)


    def forward(self, x,y, H, W, **kwargs):
        B, N, C = x.shape
        x = x.view(B, C, H, W)
        y = y.view(B, C, H, W)
        # attention
        #print(x.shape)#torch.Size([8, 32, 128, 128])
        # pad

        level_0_weight_v = self.atrous_block1(x)
        level_1_weight_v = self.atrous_block2(y)
        #print(level_0_weight_v.shape)#torch.Size([8, 32, 1, 1])
        #print(level_1_weight_v.shape)#torch.Size([8, 32, 1, 1])

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        x = x * levels_weight[:,0:1,:,:]
        y = y * levels_weight[:,1:2,:,:]
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)
        x_pad = self.pad_helper.pad_if_needed(x, x.size())
        #print(x_pad.shape)
        # permute
        x_permute = self.permute_helper.permute(x_pad, x_pad.size())
        # print(x_permute.shape)
        # torch.Size([8, 128, 128, 32])
        # torch.Size([8, 133, 133, 32])
        # torch.Size([49, 2888, 32])
        # print('------------')

        # attention
        # pad
        y_pad = self.pad_helper.pad_if_needed(y, y.size())
        # permute
        y_permute = self.permute_helper.permute(y_pad, y_pad.size())
        # attention
        out_a = self.attn(x_permute, y_permute, y_permute, **kwargs)
        # reverse permutation
        out = self.permute_helper.rev_permute(out_a, x_pad.size())
        out = self.pad_helper.depad_if_needed(out, x.size())
        return out.reshape(B, N, C)
class InterlacedPoolAttention2att(nn.Module):
    r""" interlaced sparse multi-head self attention (ISA) module with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): Window size.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, embed_dim, num_heads, window_size=7,
                 rpe=True, **kwargs):
        super(InterlacedPoolAttention2att, self).__init__()

        self.dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.with_rpe = rpe
        #print('====================================',embed_dim)
        self.attn = MHA_(embed_dim, num_heads, **kwargs)
        self.pad_helper = PadBlock(window_size)
        self.permute_helper = LocalPermuteModule(window_size)

        self.atrous_block1 = nn.Conv2d(self.dim, 8, 1, 1)
        self.atrous_block2 = nn.Conv2d(self.dim, 8, 1, 1)
        self.weight_levels = nn.Conv2d(8 * 2, 2, kernel_size=1, stride=1, padding=0)


    def forward(self, x,y, H, W, **kwargs):
        B, N, C = x.shape
        x = x.view(B, C, H, W)
        y = y.view(B, C, H, W)
        # attention
        #print(x.shape)torch.Size([8, 128, 128, 32])
        # pad

        level_0_weight_v = self.atrous_block1(x)
        level_1_weight_v = self.atrous_block2(y)


        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        x = x * levels_weight[:,0:1,:,:]
        y = y * levels_weight[:,1:2,:,:]
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)
        x_pad = self.pad_helper.pad_if_needed(x, x.size())
        #print(x_pad.shape)
        # permute
        x_permute = self.permute_helper.permute(x_pad, x_pad.size())
        # print(x_permute.shape)
        # torch.Size([8, 128, 128, 32])
        # torch.Size([8, 133, 133, 32])
        # torch.Size([49, 2888, 32])
        # print('------------')

        # attention
        # pad
        y_pad = self.pad_helper.pad_if_needed(y, y.size())
        # permute
        y_permute = self.permute_helper.permute(y_pad, y_pad.size())
        # attention
        out_a = self.attn(x_permute, y_permute, y_permute, **kwargs)
        # reverse permutation
        out = self.permute_helper.rev_permute(out_a, x_pad.size())
        out = self.pad_helper.depad_if_needed(out, x.size())
        return out.reshape(B, N, C),out_a
