import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_head import SegFormerHead
from . import mix_transformer
import numpy as np

from backbone.wavemlp import PATM,WaveBlock


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# class PATM(nn.Module):
#     def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mode='fc'):
#         super().__init__()
#
#         self.fc_h = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
#         self.fc_w = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
#         self.fc_c = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
#
#         self.tfc_h = nn.Conv2d(2 * dim, dim, (1, 7), stride=1, padding=(0, 7 // 2), groups=dim, bias=False)
#         self.tfc_w = nn.Conv2d(2 * dim, dim, (7, 1), stride=1, padding=(7 // 2, 0), groups=dim, bias=False)
#         self.reweight = Mlp(dim, dim // 4, dim * 3)
#         self.proj = nn.Conv2d(dim, dim, 1, 1, bias=True)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.mode = mode
#
#         if mode == 'fc':
#             self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
#             self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
#         else:
#             self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
#                                               nn.BatchNorm2d(dim), nn.ReLU())
#             self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
#                                               nn.BatchNorm2d(dim), nn.ReLU())
#
#     def forward(self, x):
#
#         B, C, H, W = x.shape
#         theta_h = self.theta_h_conv(x)
#         theta_w = self.theta_w_conv(x)
#
#         x_h = self.fc_h(x)
#         x_w = self.fc_w(x)
#         x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)
#         x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)
#
#         #         x_1=self.fc_h(x)
#         #         x_2=self.fc_w(x)
#         #         x_h=torch.cat([x_1*torch.cos(theta_h),x_2*torch.sin(theta_h)],dim=1)
#         #         x_w=torch.cat([x_1*torch.cos(theta_w),x_2*torch.sin(theta_w)],dim=1)
#
#         h = self.tfc_h(x_h)
#         w = self.tfc_w(x_w)
#         c = self.fc_c(x)
#         a = F.adaptive_avg_pool2d(h + w + c, output_size=1)
#         a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
#         x = h * a[0]
#         x = x + w * a[1]
#         x = x + c * a[2]
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


class WeTr2(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.dropout = torch.nn.Dropout2d(0.5)
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        # self.decoder = conv_head.LargeFOV(self.in_channels[-1], out_planes=self.num_classes)

        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes - 1, kernel_size=1,
                                     bias=False)
        self.classifier2 = nn.Conv2d(self.num_classes - 1, out_channels=self.num_classes - 1, kernel_size=1,
                                    bias=False)
        # self.classifier = nn.Sequential(
        #     nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes - 1, kernel_size=1,
        #               bias=False),
        #     #PATM(self.num_classes - 1, qkv_bias=False, qk_scale=None, attn_drop=0., mode='fc')
        # )
        #Mlp(512)
        self.wave = PATM(self.num_classes - 1, qkv_bias=False, qk_scale=None, attn_drop=0.,mode='fc')
        #self.wave = WaveBlock(512, mlp_ratio=4., qkv_bias=False, qk_scale=None,
         #         attn_drop=0., drop_path=0., norm_layer=nn.BatchNorm2d, mode='fc')
    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        # for param in list(self.classifier.parameters()):
        #     param_groups[2].append(param)
        for param in list(self.wave.parameters()):
            param_groups[2].append(param)
        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.attn_proj.weight)
        param_groups[2].append(self.attn_proj.bias)
        param_groups[2].append(self.classifier2.weight)



        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups


    # def forward(self, x, cam_only=False, seg_detach=True, affine = False,  ):
    #
    #     _x, _attns = self.encoder(x)
    #     _x1, _x2, _x3, _x4 = _x
    #
    #     seg = self.decoder(_x)
    #     # seg = self.decoder(_x4)
    #     # print('--------')
    #
    #     attn_cat = torch.cat(_attns[-2:], dim=1)  # .detach()
    #     attn_pred = self.attn_proj(attn_cat)
    #     attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]
    #     #print('_x4',_x4.shape)
    #     #cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
    #     #print('cam_s4',cam_s4.shape)
    #     #cam_s4 = self.wave(cam_s4)
    #     #print(cam_s4.shape)
    #     # _x4 torch.Size([4, 512, 10, 10])
    #     # cam_s4torch.Size([4, 20, 10, 10])
    #     # torch.Size([4, 20, 10, 10])
    #     #.load_state_dict(state['xxx_weights'])
    #     #cam_s4 = self.classifier(_x4)
    #     #cam_s4 = self.wave(cam_s4)
    #
    #
    #     if cam_only:
    #         cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
    #         return cam_s4, attn_pred
    #     wave = self.wave(_x4)
    #     # _x4 = self.dropout(_x4.clone()
    #     cls_x4 = self.pooling(wave, (1, 1))
    #     #cls_x4 = cls_x4.clone()
    #     #.detach()#.clone()#.detach()
    #     cls_x4 = self.classifier(cls_x4)
    #
    #     # if cam_only:
    #     #     cam_s4 = self.classifier(_x4).detach() #F.conv2d(_x4, self.classifier.weight).detach()
    #     #     cam_s4 =self.wave(cam_s4)
    #     #     return cam_s4, attn_pred
    #     # cls_x4 = self.wave(cls_x4)
    #     # if cam_only:
    #     #     #cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
    #     #     return cls_x4, attn_pred
    #
    #     cls_x4 = cls_x4.view(-1, self.num_classes - 1)
    #
    #     if affine:
    #         return cls_x4, seg, _attns
    #
    #
    #     return cls_x4, seg, _attns, attn_pred

    #2
    def forward(self, x, cam_only=False, seg_detach=True, affine = False,  ):

        _x, _attns = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x

        seg = self.decoder(_x)
        # seg = self.decoder(_x4)
        # print('--------')

        attn_cat = torch.cat(_attns[-2:], dim=1)  # .detach()
        attn_pred = self.attn_proj(attn_cat)
        attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]
        #print('_x4',_x4.shape)
        #cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
        #print('cam_s4',cam_s4.shape)
        #cam_s4 = self.wave(cam_s4)
        #print(cam_s4.shape)
        # _x4 torch.Size([4, 512, 10, 10])
        # cam_s4torch.Size([4, 20, 10, 10])
        # torch.Size([4, 20, 10, 10])
        #.load_state_dict(state['xxx_weights'])
        #cam_s4 = self.classifier(_x4)
        #cam_s4 = self.wave(cam_s4)




        # _x4 = self.dropout(_x4.clone()
        cls_x4 = self.pooling(_x4, (1, 1))
        #cls_x4 = cls_x4.clone()
        #.detach()#.clone()#.detach()
        cls_x4 = self.classifier(cls_x4)
        wave = self.wave(cls_x4)
        cls_x4 = self.classifier2(wave)
        if cam_only:
            cam_s4 = F.conv2d(wave, self.classifier2.weight).detach()
            return cam_s4, attn_pred
        # if cam_only:
        #     cam_s4 = self.classifier(_x4).detach() #F.conv2d(_x4, self.classifier.weight).detach()
        #     cam_s4 =self.wave(cam_s4)
        #     return cam_s4, attn_pred
        # cls_x4 = self.wave(cls_x4)
        # if cam_only:
        #     #cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
        #     return cls_x4, attn_pred

        cls_x4 = cls_x4.view(-1, self.num_classes - 1)

        if affine:
            return cls_x4, seg, _attns


        return cls_x4, seg, _attns, attn_pred


# def forward(self, x, cam_only=False, seg_detach=True, ):
    #
    #     _x, _attns = self.encoder(x)
    #     _x1, _x2, _x3, _x4 = _x
    #
    #     seg = self.decoder(_x)
    #     # seg = self.decoder(_x4)
    #
    #     attn_cat = torch.cat(_attns[-2:], dim=1)  # .detach()
    #     attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)
    #     attn_pred = self.attn_proj(attn_cat)
    #     attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]
    #
    #     if cam_only:
    #         cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
    #         return cam_s4, attn_pred
    #
    #     # _x4 = self.dropout(_x4.clone()
    #     cls_x4 = self.pooling(_x4, (1, 1))
    #     cls_x4 = self.classifier(cls_x4)
    #     cls_x4 = cls_x4.view(-1, self.num_classes - 1)
    #
    #     # attns = [attn[:,0,...] for attn in _attns]
    #     # attns.append(attn_pred)
    #     return cls_x4, seg, _attns


    # def forward(self, x, cam_only=False, seg_detach=True, ):
    #
    #     _x, _attns = self.encoder(x)
    #     _x1, _x2, _x3, _x4 = _x
    #
    #     seg = self.decoder(_x)
    #     # seg = self.decoder(_x4)
    #
    #     attn_cat = torch.cat(_attns[-2:], dim=1)  # .detach()
    #     attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)
    #     attn_pred = self.attn_proj(attn_cat)
    #     attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]
    #
    #     if cam_only:
    #         cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
    #         return cam_s4, attn_pred
    #
    #     # _x4 = self.dropout(_x4.clone()
    #     cls_x4 = self.pooling(_x4, (1, 1))
    #     cls_x4 = self.classifier(cls_x4)
    #     cls_x4 = cls_x4.view(-1, self.num_classes - 1)
    #
    #     # attns = [attn[:,0,...] for attn in _attns]
    #     # attns.append(attn_pred)
    #     return cls_x4, seg, _attns, attn_pred

class WeTr599(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.dropout = torch.nn.Dropout2d(0.5)
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        # self.decoder = conv_head.LargeFOV(self.in_channels[-1], out_planes=self.num_classes)
        self.attn_proj1 = nn.Conv2d(in_channels=512, out_channels=8, kernel_size=1, bias=True)
        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes - 1, kernel_size=1,
                                    bias=False)
        self.classifier2 = nn.Conv2d(self.num_classes - 1, out_channels=self.num_classes - 1, kernel_size=1,
                                     bias=False)
        # self.classifier = nn.Sequential(
        #     nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes - 1, kernel_size=1,
        #               bias=False),
        #     #PATM(self.num_classes - 1, qkv_bias=False, qk_scale=None, attn_drop=0., mode='fc')
        # )
        # Mlp(512)
        self.wave = PATM(512, qkv_bias=False, qk_scale=None, attn_drop=0., mode='fc')
        # self.wave = WaveBlock(512, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        #         attn_drop=0., drop_path=0., norm_layer=nn.BatchNorm2d, mode='fc')

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        # for param in list(self.classifier.parameters()):
        #     param_groups[2].append(param)
        for param in list(self.wave.parameters()):
            param_groups[2].append(param)
        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.attn_proj.weight)
        param_groups[2].append(self.attn_proj.bias)
        param_groups[2].append(self.attn_proj1.weight)
        param_groups[2].append(self.attn_proj1.bias)
        param_groups[2].append(self.classifier2.weight)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    # 1
    def forward(self, x, cam_only=False, seg_detach=True, affine=False, ):
        #print(x.shape)torch.Size([2, 3, 320, 320])
        _x, _attns = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x
        #print('_x4', _x4.shape)_x4 torch.Size([2, 512, 20, 20])
        #_x4 = self.wave(_x4)
        #_x = torch.cat(_x1, _x2, _x3, _x4)
        seg = self.decoder(_x)
        # seg = self.decoder(_x4)
        # print('--------')
        # print('_attns[-1]', _attns[-1].shape)
        # print('_attns[-2]',_attns[-2].shape)
        # _attns[-1] torch.Size([2, 8, 400, 400])
        # _attns[-2] torch.Size([2, 8, 400, 400])
        #_x4a = _x4.reshape(_x4.shape[0], 8, 160,160)

        #print('_x4a', _x4a.shape)#_x4a torch.Size([8, 8, 400, 400])
        #attn_cat = torch.cat(_attns[-2:], dim=1)  # .detach()
        #print('attn_cat', attn_cat.shape)
        _x4a = self.attn_proj1(_x4)
        _x4a = F.interpolate(_x4a, size=(_attns[-1].shape[3], _attns[-1].shape[3]), mode='bilinear',
                            align_corners=True)
        attn_cat = torch.cat((_attns[-1],_x4a), dim=1)
        attn_pred = self.attn_proj(attn_cat)
        #print( attn_pred.shape)#[2, 1, 400, 400])
        attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]
        # print('_x4',_x4.shape)
        # cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
        # print('cam_s4',cam_s4.shape)
        # cam_s4 = self.wave(cam_s4)
        # print(cam_s4.shape)
        # _x4 torch.Size([4, 512, 10, 10])
        # cam_s4torch.Size([4, 20, 10, 10])
        # torch.Size([4, 20, 10, 10])
        # .load_state_dict(state['xxx_weights'])
        # cam_s4 = self.classifier(_x4)
        # cam_s4 = self.wave(cam_s4)

        # wave = self.wave(_x4) 0.03
        # # _x4 = self.dropout(_x4.clone()
        # wave = self.pooling(wave, (1, 1))
        # # cls_x4 = cls_x4.clone()
        # # .detach()#.clone()#.detach()
        # cls_x4 = self.classifier(wave)
        #
        # wave = self.wave(_x4) #0.37
        # cls_x4 = self.pooling(wave, (1, 1))
        # cls_x4 = self.classifier(cls_x4)  #wave


        # wave = self.wave(_x4) #0.12 0.15
        # cls_x4 = self.pooling(wave, (1, 1))
        # cls_x4 = self.classifier(cls_x4)  #_x4

        # 0.03
        # cls_x4 = self.pooling(_x4, (1, 1))
        # wave = self.wave(cls_x4)
        # cls_x4 = self.classifier(wave)
        # if cam_only:
        #     cam_s4 = F.conv2d(wave, self.classifier.weight).detach()

        # cls_x4 = self.pooling(_x4,(1,1))
        # cls_x4 = self.classifier(cls_x4)
        # cls_x4 = cls_x4.view(-1, self.num_classes-1)
        # if cam_only:
        #     cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
        #     return cam_s4, attn_pred


        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)
        if cam_only:
            cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
            return cam_s4, attn_pred


        # if cam_only:
        #     cam_s4 = self.classifier(_x4).detach() #F.conv2d(_x4, self.classifier.weight).detach()
        #     cam_s4 =self.wave(cam_s4)
        #     return cam_s4, attn_pred
        # cls_x4 = self.wave(cls_x4)
        # if cam_only:
        #     #cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
        #     return cls_x4, attn_pred

        cls_x4 = cls_x4.view(-1, self.num_classes - 1)

        if affine:
            return cls_x4, seg, _attns

        return cls_x4, seg, _attns, attn_pred


        # def forward(self, x, cam_only=False, seg_detach=True, ):
        #
        #     _x, _attns = self.encoder(x)
        #     _x1, _x2, _x3, _x4 = _x
        #
        #     seg = self.decoder(_x)
        #     # seg = self.decoder(_x4)
        #
        #     attn_cat = torch.cat(_attns[-2:], dim=1)  # .detach()
        #     attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)
        #     attn_pred = self.attn_proj(attn_cat)
        #     attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]
        #
        #     if cam_only:
        #         cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
        #         return cam_s4, attn_pred
        #
        #     # _x4 = self.dropout(_x4.clone()
        #     cls_x4 = self.pooling(_x4, (1, 1))
        #     cls_x4 = self.classifier(cls_x4)
        #     cls_x4 = cls_x4.view(-1, self.num_classes - 1)
        #
        #     # attns = [attn[:,0,...] for attn in _attns]
        #     # attns.append(attn_pred)
        #     return cls_x4, seg, _attns


        # def forward(self, x, cam_only=False, seg_detach=True, ):
        #
        #     _x, _attns = self.encoder(x)
        #     _x1, _x2, _x3, _x4 = _x
        #
        #     seg = self.decoder(_x)
        #     # seg = self.decoder(_x4)
        #
        #     attn_cat = torch.cat(_attns[-2:], dim=1)  # .detach()
        #     attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)
        #     attn_pred = self.attn_proj(attn_cat)
        #     attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]
        #
        #     if cam_only:
        #         cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
        #         return cam_s4, attn_pred
        #
        #     # _x4 = self.dropout(_x4.clone()
        #     cls_x4 = self.pooling(_x4, (1, 1))
        #     cls_x4 = self.classifier(cls_x4)
        #     cls_x4 = cls_x4.view(-1, self.num_classes - 1)
        #
        #     # attns = [attn[:,0,...] for attn in _attns]
        #     # attns.append(attn_pred)
        #     return cls_x4, seg, _attns, attn_pred
class WeTr522(nn.Module):
    def forward(self, x, cam_only=False, seg_detach=True, affine=False, ):
        #print(x.shape)torch.Size([2, 3, 320, 320])
        _x, _attns = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x
        #print('_x4', _x4.shape)_x4 torch.Size([2, 512, 20, 20])
        #_x4 = self.wave(_x4)
        #_x = torch.cat(_x1, _x2, _x3, _x4)
        seg = self.decoder(_x)
        _x4a = self.attn_proj1(_x4)
        _x4a = F.interpolate(_x4a, size=(_attns[-1].shape[3], _attns[-1].shape[3]), mode='bilinear',
                            align_corners=True)
        attn_pred = self.attn_proj(_x4a)
        attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]
        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)
        if cam_only:
            cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
            return cam_s4, attn_pred

        cls_x4 = cls_x4.view(-1, self.num_classes - 1)
        if affine:
            return cls_x4, seg, _attns
        return cls_x4, seg, _attns, attn_pred

class WeTr603(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.dropout = torch.nn.Dropout2d(0.5)
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        # self.decoder = conv_head.LargeFOV(self.in_channels[-1], out_planes=self.num_classes)
        self.attn_proj1 = nn.Conv2d(in_channels=512, out_channels=8, kernel_size=1, bias=True)
        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes - 1, kernel_size=1,
                                    bias=False)
        self.classifier2 = nn.Conv2d(self.num_classes - 1, out_channels=self.num_classes - 1, kernel_size=1,
                                     bias=False)

        self.wave = PATM(512, qkv_bias=False, qk_scale=None, attn_drop=0., mode='fc')
        # self.wave = WaveBlock(512, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        #         attn_drop=0., drop_path=0., norm_layer=nn.BatchNorm2d, mode='fc')

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        # for param in list(self.classifier.parameters()):
        #     param_groups[2].append(param)
        for param in list(self.wave.parameters()):
            param_groups[2].append(param)
        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.attn_proj.weight)
        param_groups[2].append(self.attn_proj.bias)
        param_groups[2].append(self.attn_proj1.weight)
        param_groups[2].append(self.attn_proj1.bias)
        param_groups[2].append(self.classifier2.weight)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    # 1
    def forward(self, x, cam_only=False, seg_detach=True, affine=False, ):
        #print(x.shape)torch.Size([2, 3, 320, 320])
        _x, _attns = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x
        #print('_x4', _x4.shape)_x4 torch.Size([2, 512, 20, 20])
        #_x4 = self.wave(_x4)
        #_x = torch.cat(_x1, _x2, _x3, _x4)
        seg = self.decoder(_x)

        _x4a = self.attn_proj1(_x4)
        _x4a = F.interpolate(_x4a, size=(_attns[-1].shape[3], _attns[-1].shape[3]), mode='bilinear',
                            align_corners=True)
        attn_cat = torch.cat((_attns[-2],_x4a), dim=1)
        attn_pred = self.attn_proj(attn_cat)
        #print( attn_pred.shape)#[2, 1, 400, 400])
        attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]
        # print('_x4',_x4.shape)
        # cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
        # print('cam_s4',cam_s4.shape)
        # cam_s4 = self.wave(cam_s4)
        # print(cam_s4.shape)
        # _x4 torch.Size([4, 512, 10, 10])
        # cam_s4torch.Size([4, 20, 10, 10])
        # torch.Size([4, 20, 10, 10])
        # .load_state_dict(state['xxx_weights'])
        # cam_s4 = self.classifier(_x4)
        # cam_s4 = self.wave(cam_s4)

        # wave = self.wave(_x4) 0.03
        # # _x4 = self.dropout(_x4.clone()
        # wave = self.pooling(wave, (1, 1))
        # # cls_x4 = cls_x4.clone()
        # # .detach()#.clone()#.detach()
        # cls_x4 = self.classifier(wave)
        #
        # wave = self.wave(_x4) #0.37
        # cls_x4 = self.pooling(wave, (1, 1))
        # cls_x4 = self.classifier(cls_x4)  #wave


        # wave = self.wave(_x4) #0.12 0.15
        # cls_x4 = self.pooling(wave, (1, 1))
        # cls_x4 = self.classifier(cls_x4)  #_x4

        # 0.03
        # cls_x4 = self.pooling(_x4, (1, 1))
        # wave = self.wave(cls_x4)
        # cls_x4 = self.classifier(wave)
        # if cam_only:
        #     cam_s4 = F.conv2d(wave, self.classifier.weight).detach()

        # cls_x4 = self.pooling(_x4,(1,1))
        # cls_x4 = self.classifier(cls_x4)
        # cls_x4 = cls_x4.view(-1, self.num_classes-1)
        # if cam_only:
        #     cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
        #     return cam_s4, attn_pred


        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)
        if cam_only:
            cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
            return cam_s4, attn_pred


        # if cam_only:
        #     cam_s4 = self.classifier(_x4).detach() #F.conv2d(_x4, self.classifier.weight).detach()
        #     cam_s4 =self.wave(cam_s4)
        #     return cam_s4, attn_pred
        # cls_x4 = self.wave(cls_x4)
        # if cam_only:
        #     #cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
        #     return cls_x4, attn_pred

        cls_x4 = cls_x4.view(-1, self.num_classes - 1)

        if affine:
            return cls_x4, seg, _attns

        return cls_x4, seg, _attns, attn_pred


        # def forward(self, x, cam_only=False, seg_detach=True, ):
        #
        #     _x, _attns = self.encoder(x)
        #     _x1, _x2, _x3, _x4 = _x
        #
        #     seg = self.decoder(_x)
        #     # seg = self.decoder(_x4)
        #
        #     attn_cat = torch.cat(_attns[-2:], dim=1)  # .detach()
        #     attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)
        #     attn_pred = self.attn_proj(attn_cat)
        #     attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]
        #
        #     if cam_only:
        #         cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
        #         return cam_s4, attn_pred
        #
        #     # _x4 = self.dropout(_x4.clone()
        #     cls_x4 = self.pooling(_x4, (1, 1))
        #     cls_x4 = self.classifier(cls_x4)
        #     cls_x4 = cls_x4.view(-1, self.num_classes - 1)
        #
        #     # attns = [attn[:,0,...] for attn in _attns]
        #     # attns.append(attn_pred)
        #     return cls_x4, seg, _attns


        # def forward(self, x, cam_only=False, seg_detach=True, ):
        #
        #     _x, _attns = self.encoder(x)
        #     _x1, _x2, _x3, _x4 = _x
        #
        #     seg = self.decoder(_x)
        #     # seg = self.decoder(_x4)
        #
        #     attn_cat = torch.cat(_attns[-2:], dim=1)  # .detach()
        #     attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)
        #     attn_pred = self.attn_proj(attn_cat)
        #     attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]
        #
        #     if cam_only:
        #         cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
        #         return cam_s4, attn_pred
        #
        #     # _x4 = self.dropout(_x4.clone()
        #     cls_x4 = self.pooling(_x4, (1, 1))
        #     cls_x4 = self.classifier(cls_x4)
        #     cls_x4 = cls_x4.view(-1, self.num_classes - 1)
        #
        #     # attns = [attn[:,0,...] for attn in _attns]
        #     # attns.append(attn_pred)
        #     return cls_x4, seg, _attns, attn_pred

class WeTr(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.dropout = torch.nn.Dropout2d(0.5)
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        # self.decoder = conv_head.LargeFOV(self.in_channels[-1], out_planes=self.num_classes)
        self.attn_proj1 = nn.Conv2d(in_channels=512, out_channels=8, kernel_size=1, bias=True)
        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes - 1, kernel_size=1,
                                    bias=False)
        self.classifier2 = nn.Conv2d(self.num_classes - 1, out_channels=self.num_classes - 1, kernel_size=1,
                                     bias=False)

        self.wave = PATM(512, qkv_bias=False, qk_scale=None, attn_drop=0., mode='fc')
        # self.wave = WaveBlock(512, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        #         attn_drop=0., drop_path=0., norm_layer=nn.BatchNorm2d, mode='fc')

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        # for param in list(self.classifier.parameters()):
        #     param_groups[2].append(param)
        for param in list(self.wave.parameters()):
            param_groups[2].append(param)
        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.attn_proj.weight)
        param_groups[2].append(self.attn_proj.bias)
        param_groups[2].append(self.attn_proj1.weight)
        param_groups[2].append(self.attn_proj1.bias)
        param_groups[2].append(self.classifier2.weight)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    # 1
    def forward(self, x, cam_only=False, seg_detach=True, affine=False, ):
        #print(x.shape)torch.Size([2, 3, 320, 320])
        _x, _attns = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x
        #print('_x4', _x4.shape)_x4 torch.Size([2, 512, 20, 20])
        _x4 = self.wave(_x4)
        #_x = torch.cat(_x1, _x2, _x3, _x4)
        seg = self.decoder(_x)

        _x4a = self.attn_proj1(_x4)
        _x4a = F.interpolate(_x4a, size=(_attns[-1].shape[3], _attns[-1].shape[3]), mode='bilinear',
                            align_corners=True)
        attn_cat = torch.cat((_attns[-2],_x4a), dim=1)
        attn_pred = self.attn_proj(attn_cat)
        #print( attn_pred.shape)#[2, 1, 400, 400])
        attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]
        # print('_x4',_x4.shape)
        # cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
        # print('cam_s4',cam_s4.shape)
        # cam_s4 = self.wave(cam_s4)
        # print(cam_s4.shape)
        # _x4 torch.Size([4, 512, 10, 10])
        # cam_s4torch.Size([4, 20, 10, 10])
        # torch.Size([4, 20, 10, 10])
        # .load_state_dict(state['xxx_weights'])
        # cam_s4 = self.classifier(_x4)
        # cam_s4 = self.wave(cam_s4)

        # wave = self.wave(_x4) 0.03
        # # _x4 = self.dropout(_x4.clone()
        # wave = self.pooling(wave, (1, 1))
        # # cls_x4 = cls_x4.clone()
        # # .detach()#.clone()#.detach()
        # cls_x4 = self.classifier(wave)
        #
        # wave = self.wave(_x4) #0.37
        # cls_x4 = self.pooling(wave, (1, 1))
        # cls_x4 = self.classifier(cls_x4)  #wave


        # wave = self.wave(_x4) #0.12 0.15
        # cls_x4 = self.pooling(wave, (1, 1))
        # cls_x4 = self.classifier(cls_x4)  #_x4

        # 0.03
        # cls_x4 = self.pooling(_x4, (1, 1))
        # wave = self.wave(cls_x4)
        # cls_x4 = self.classifier(wave)
        # if cam_only:
        #     cam_s4 = F.conv2d(wave, self.classifier.weight).detach()

        # cls_x4 = self.pooling(_x4,(1,1))
        # cls_x4 = self.classifier(cls_x4)
        # cls_x4 = cls_x4.view(-1, self.num_classes-1)
        # if cam_only:
        #     cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
        #     return cam_s4, attn_pred


        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)
        if cam_only:
            cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
            return cam_s4, attn_pred


        # if cam_only:
        #     cam_s4 = self.classifier(_x4).detach() #F.conv2d(_x4, self.classifier.weight).detach()
        #     cam_s4 =self.wave(cam_s4)
        #     return cam_s4, attn_pred
        # cls_x4 = self.wave(cls_x4)
        # if cam_only:
        #     #cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
        #     return cls_x4, attn_pred

        cls_x4 = cls_x4.view(-1, self.num_classes - 1)

        if affine:
            return cls_x4, seg, _attns

        return cls_x4, seg, _attns, attn_pred


        # def forward(self, x, cam_only=False, seg_detach=True, ):
        #
        #     _x, _attns = self.encoder(x)
        #     _x1, _x2, _x3, _x4 = _x
        #
        #     seg = self.decoder(_x)
        #     # seg = self.decoder(_x4)
        #
        #     attn_cat = torch.cat(_attns[-2:], dim=1)  # .detach()
        #     attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)
        #     attn_pred = self.attn_proj(attn_cat)
        #     attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]
        #
        #     if cam_only:
        #         cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
        #         return cam_s4, attn_pred
        #
        #     # _x4 = self.dropout(_x4.clone()
        #     cls_x4 = self.pooling(_x4, (1, 1))
        #     cls_x4 = self.classifier(cls_x4)
        #     cls_x4 = cls_x4.view(-1, self.num_classes - 1)
        #
        #     # attns = [attn[:,0,...] for attn in _attns]
        #     # attns.append(attn_pred)
        #     return cls_x4, seg, _attns


        # def forward(self, x, cam_only=False, seg_detach=True, ):
        #
        #     _x, _attns = self.encoder(x)
        #     _x1, _x2, _x3, _x4 = _x
        #
        #     seg = self.decoder(_x)
        #     # seg = self.decoder(_x4)
        #
        #     attn_cat = torch.cat(_attns[-2:], dim=1)  # .detach()
        #     attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)
        #     attn_pred = self.attn_proj(attn_cat)
        #     attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]
        #
        #     if cam_only:
        #         cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
        #         return cam_s4, attn_pred
        #
        #     # _x4 = self.dropout(_x4.clone()
        #     cls_x4 = self.pooling(_x4, (1, 1))
        #     cls_x4 = self.classifier(cls_x4)
        #     cls_x4 = cls_x4.view(-1, self.num_classes - 1)
        #
        #     # attns = [attn[:,0,...] for attn in _attns]
        #     # attns.append(attn_pred)
        #     return cls_x4, seg, _attns, attn_pred

if __name__=="__main__":

    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    wetr = WeTr('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
    wetr._param_groups()
    dummy_input = torch.rand(2,3,512,512)
    wetr(dummy_input)