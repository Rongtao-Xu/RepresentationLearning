import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_head import SegFormerHead
from . import mix_transformer
import numpy as np


class TSCD(nn.Module):
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
        #self.decoder = SegFormerHead(feature_strides=self.feature_strides, #in_channels=self.in_channels,
        #                             embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        #self.decoder = conv_head.LargeFOV(self.in_channels[-1], out_planes=self.num_classes)

        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes - 1, kernel_size=1,
                                    bias=False)

        self.head = nn.Conv2d(in_channels=512, out_channels=20, kernel_size=1, bias=True)
        self.neck = SimpleFusion8(1024)
    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.attn_proj.weight)
        param_groups[2].append(self.attn_proj.bias)

        # for param in list(self.decoder.parameters()):
        #     param_groups[3].append(param)

        return param_groups

    def forward(self, x, cam_only=False, seg_detach=True, aux = False,  ):

        _x, _attns = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x
        x4 = self.neck(_x)

        #_c4 = F.interpolate(_x4, size=_x3.size()[2:], mode='bilinear', align_corners=True)
        #_c3 = F.interpolate(_x3, size=_x2.size()[2:], mode='bilinear', align_corners=True)
        #print(_x4.shape)
        # print(_x3.shape)
        # torch.Size([2, 512, 20, 20])
        # torch.Size([2, 320, 20, 20])

        seg = x4  #+ _x3 #self.decoder(_x)

        attn_cat = torch.cat(_attns[-2:], dim=1)  # .detach()
        attn_pred = self.attn_proj(attn_cat)
        attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]
        # print(attn_pred.shape)
        if cam_only:
            cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
            return cam_s4, attn_pred

        # _x4 = self.dropout(_x4.clone()
        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes - 1)

        if aux:
            return cls_x4, seg, _attns

        return cls_x4, seg, _attns, attn_pred

class SimpleFusion8(nn.Module):
    def __init__(self, in_channels):
        super(SimpleFusion8, self).__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, 15, 1),
            nn.BatchNorm2d(15),
            nn.ReLU(True)
        )

    def forward(self, feat_list):
        # print(feat_list[0].shape)
        # print(feat_list[1].shape)
        # print(feat_list[2].shape)
        # print(feat_list[3].shape)
        # torch.Size([2, 64, 80, 80])
        # torch.Size([2, 128, 40, 40])
        # torch.Size([2, 320, 20, 20])
        # torch.Size([2, 512, 20, 20])

        x0 = feat_list[0]
        x0_h, x0_w = x0.size(2), x0.size(3)
        x1 = F.interpolate(feat_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(feat_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(feat_list[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = self.fuse_conv(x)
        #print(x.shape)
        return x


class Class_Predictor(nn.Module):
    def __init__(self, num_classes, representation_size):
        super(Class_Predictor, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Conv2d(representation_size, num_classes, 1, bias=False)

    def forward(self, x, label):
        batch_size = x.shape[0]
        #print(label.shape)torch.Size([1, 20])
        x = x.reshape(batch_size, self.num_classes, -1)  # bs*20*2048
        mask = label > 0  # bs*20
        #print(mask.shape)
        #print(x.shape)torch.Size([2, 20, 512])
        feature_list = [x[i][mask[i]] for i in range(2)]  # bs*n*2048
        prediction = [self.classifier(y.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1) for y in feature_list]
        labels = [torch.nonzero(label[i]).squeeze(1) for i in range(label.shape[0])]

        loss = 0
        acc = 0
        num = 0
        for logit, label in zip(prediction, labels):
            if label.shape[0] == 0:
                continue
            loss_ce = F.cross_entropy(logit, label)
            loss += loss_ce
            acc += (logit.argmax(dim=1) == label.view(-1)).sum().float()
            num += label.size(0)

        return loss / batch_size, acc / num

if __name__=="__main__":

    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    wetr = WeTr('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
    wetr._param_groups()
    dummy_input = torch.rand(2,3,512,512)
    wetr(dummy_input)