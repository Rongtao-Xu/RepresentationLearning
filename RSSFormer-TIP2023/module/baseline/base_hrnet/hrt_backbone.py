
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url

BatchNorm2d = nn.BatchNorm2d

BN_MOMENTUM = 0.1

__all__ = ['HighResolutionNet', 'HighResolutionModule',
           'hrnetv2_w18', 'hrnetv2_w32', 'hrnetv2_w40', 'hrnetv2_w48']

model_urls = {
    "hrnetv2_w18": "https://download.openmmlab.com/pretrain/third_party/hrnetv2_w18-00eb2006.pth",
    "hrnetv2_w32": "https://download.openmmlab.com/pretrain/third_party/hrnetv2_w32-dc9eeb4f.pth",
    "hrnetv2_w40": "https://download.openmmlab.com/pretrain/third_party/hrnetv2_w40-ed0b031c.pth",
    "hrnetv2_w48": "https://download.openmmlab.com/pretrain/third_party/hrnetv2_w48-d2186c55.pth",
}

def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)







import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.bottleneck_block import BottleneckDWP
from .modules.transformer_block import GeneralTransformerBlock

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



blocks_dict = {
    "BOTTLENECK": Bottleneck,
    "TRANSFORMER_BLOCK": GeneralTransformerBlock,
}


BN_MOMENTUM = 0.1


class HighResolutionTransformerModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        multi_scale_output=True,
        drop_path=0.0,
    ):
        """Based on Local-Attention & FFN-DW-BN
        num_heads: the number of head witin each MHSA
        num_window_sizes: the window size for the local self-attention
        num_halo_sizes: the halo size around the local window
            - reference: ``Scaling Local Self-Attention for Parameter Efficient Visual Backbones''
        num_sr_ratios: the spatial reduction ratios of PVT/SRA scheme.
            - reference: ``Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions''
        """
        super(HighResolutionTransformerModule, self).__init__()


        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            drop_path,
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

        self.num_heads = num_heads
        self.num_window_sizes = num_window_sizes
        self.num_mlp_ratios = num_mlp_ratios


    def _make_one_branch(
        self,
        branch_index,
        block,
        num_blocks,
        num_channels,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        drop_paths,
        stride=1,
    ):
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.SyncBatchNorm(
                    num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                num_heads=num_heads[branch_index],
                window_size=num_window_sizes[branch_index],
                mlp_ratio=num_mlp_ratios[branch_index],
                drop_path=drop_paths[0],
            )
        )

        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    num_heads=num_heads[branch_index],
                    window_size=num_window_sizes[branch_index],
                    mlp_ratio=num_mlp_ratios[branch_index],
                    drop_path=drop_paths[i],
                )
            )
        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches,
        block,
        num_blocks,
        num_channels,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        drop_paths,
    ):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    drop_paths=[_ * (2 ** i) for _ in drop_paths]
                    if os.environ.get("multi_res_drop_path", False)
                    else drop_paths,
                )
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                kernel_size=1,
                                stride=1,
                                bias=False,
                            ),
                            nn.SyncBatchNorm(num_inchannels[i], momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_inchannels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=num_inchannels[j],
                                        bias=False,
                                    ),
                                    nn.SyncBatchNorm(
                                        num_inchannels[j], momentum=BN_MOMENTUM
                                    ),
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        kernel_size=1,
                                        stride=1,
                                        bias=False,
                                    ),
                                    nn.SyncBatchNorm(
                                        num_outchannels_conv3x3, momentum=BN_MOMENTUM
                                    ),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_inchannels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=num_inchannels[j],
                                        bias=False,
                                    ),
                                    nn.SyncBatchNorm(
                                        num_inchannels[j], momentum=BN_MOMENTUM
                                    ),
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        kernel_size=1,
                                        stride=1,
                                        bias=False,
                                    ),
                                    nn.SyncBatchNorm(
                                        num_outchannels_conv3x3, momentum=BN_MOMENTUM
                                    ),
                                    nn.ReLU(False),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode="bilinear",
                        align_corners=True,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HighResolutionTransformer(nn.Module):
    def __init__(self, cfg, norm_eval,**kwargs):
        super(HighResolutionTransformer, self).__init__()
        self.norm_eval = norm_eval
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.SyncBatchNorm(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.SyncBatchNorm(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # stochastic depth
        depth_s2 = cfg["STAGE2"]["NUM_BLOCKS"][0] * cfg["STAGE2"]["NUM_MODULES"]
        depth_s3 = cfg["STAGE3"]["NUM_BLOCKS"][0] * cfg["STAGE3"]["NUM_MODULES"]
        depth_s4 = cfg["STAGE4"]["NUM_BLOCKS"][0] * cfg["STAGE4"]["NUM_MODULES"]
        depths = [depth_s2, depth_s3, depth_s4]
        drop_path_rate = cfg["DROP_PATH_RATE"]
        if os.environ.get("drop_path_rate") is not None:
            drop_path_rate = float(os.environ.get("drop_path_rate"))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stage1_cfg = cfg["STAGE1"]
        num_channels = self.stage1_cfg["NUM_CHANNELS"][0]
        block = blocks_dict[self.stage1_cfg["BLOCK"]]
        num_blocks = self.stage1_cfg["NUM_BLOCKS"][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = cfg["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels
        )
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, drop_path=dpr[0:depth_s2]
        )

        self.stage3_cfg = cfg["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, drop_path=dpr[depth_s2 : depth_s2 + depth_s3]
        )

        self.stage4_cfg = cfg["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multi_scale_output=True,
            drop_path=dpr[depth_s2 + depth_s3 :],
        )

        if os.environ.get("keep_imagenet_head"):
            (
                self.incre_modules,
                self.downsamp_modules,
                self.final_layer,
            ) = self._make_head(pre_stage_channels)

    def _make_head(self, pre_stage_channels):
        head_block = BottleneckDWP
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(
                head_block, channels, head_channels[i], 1, stride=1
            )
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_channels,
                ),
                nn.SyncBatchNorm(in_channels, momentum=BN_MOMENTUM),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.SyncBatchNorm(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.SyncBatchNorm(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.SyncBatchNorm(
                                num_channels_cur_layer[i], momentum=BN_MOMENTUM
                            ),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.SyncBatchNorm(outchannels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(
        self,
        block,
        inplanes,
        planes,
        blocks,
        num_heads=1,
        stride=1,
        window_size=7,
        mlp_ratio=4.0,
    ):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.SyncBatchNorm(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []

        if isinstance(block, GeneralTransformerBlock):
            layers.append(
                block(
                    inplanes,
                    planes,
                    num_heads,
                    window_size,
                    mlp_ratio,
                )
            )
        else:
            layers.append(block(inplanes, planes, stride, downsample))

        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(
        self, layer_config, num_inchannels, multi_scale_output=True, drop_path=0.0
    ):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        num_heads = layer_config["NUM_HEADS"]
        num_window_sizes = layer_config["NUM_WINDOW_SIZES"]
        num_mlp_ratios = layer_config["NUM_MLP_RATIOS"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionTransformerModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    reset_multi_scale_output,
                    drop_path=drop_path[num_blocks[0] * i : num_blocks[0] * (i + 1)],
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        if os.environ.get("keep_imagenet_head"):
            x_list = []
            y = self.incre_modules[0](y_list[0])
            x_list.append(y)
            for i in range(len(self.downsamp_modules)):
                y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](
                    y
                )
                x_list.append(y)

            y = self.final_layer(y)
            del x_list[-1]
            x_list.append(y)
            return x_list

        else:
            return y_list

    def train(self, mode=True):
        super(HighResolutionTransformer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

 #       from lib.models.backbones.hrt.hrt_config import MODEL_CONFIGS




def hrnetv2_w32(pretrained=False, weight_path=None, norm_eval=False, frozen_stages=-1):
    from .hrt_config import MODEL_CONFIGS
    model = HighResolutionTransformer(MODEL_CONFIGS['hrt_small'],norm_eval)
    # if pretrained:
    #     if weight_path is not None:
    #         state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
    #     else:
    #         state_dict = load_state_dict_from_url(model_urls['hrnetv2_w32'],
    #                                               progress=True)
    #     model.load_state_dict(state_dict, strict=False)
    return model


