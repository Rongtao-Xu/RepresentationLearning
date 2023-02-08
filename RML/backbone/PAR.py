###
#local pixel refinement
###

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_kernel():
    
    weight = torch.zeros(8, 1, 3, 3)
    weight[0, 0, 0, 0] = 1
    weight[1, 0, 0, 1] = 1
    weight[2, 0, 0, 2] = 1

    weight[3, 0, 1, 0] = 1
    weight[4, 0, 1, 2] = 1

    weight[5, 0, 2, 0] = 1
    weight[6, 0, 2, 1] = 1
    weight[7, 0, 2, 2] = 1

    return weight


class PAR(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)

        b, c, h, w = imgs.shape
        _imgs = self.get_dilated_neighbors(imgs)
        _pos = self.pos.to(_imgs.device)

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _pos_rep = _pos.repeat(b, 1, 1, h, w)

        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)
        _pos_std = torch.std(_pos_rep, dim=self.dim, keepdim=True)

        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)

        pos_aff = -(_pos_rep / (_pos_std + 1e-8) / self.w1) ** 2
        # pos_aff = pos_aff.mean(dim=1, keepdim=True)

        aff = F.softmax(aff, dim=2) + self.w2 * F.softmax(pos_aff, dim=2)

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks

class PARgg(nn.Module):

    def __init__(self, dilations, num_iter,):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d]*4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b*c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)
 
        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)
        
        for d in self.dilations:
            pos_xy.append(ker*d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)

        imgs2 = F.interpolate(imgs, scale_factor=0.3, mode='bilinear', align_corners=True)
        b, c, h, w = imgs2.shape
        _imgs2 = self.get_dilated_neighbors(imgs2)
        _imgs_rep2 = imgs2.unsqueeze(self.dim).repeat(1, 1, _imgs2.shape[self.dim], 1, 1)
        #print('_imgs2', _imgs2.shape)
        #print('_imgs_rep2', _imgs_rep2.shape)
        _imgs_abs2 = torch.abs(_imgs2 - _imgs_rep2)
        _imgs_std2 = torch.std(_imgs2, dim=self.dim, keepdim=True)
        aff2 = -(_imgs_abs2 / (_imgs_std2 + 1e-8) ) ** 2
        aff2 = aff2.mean(dim=1, keepdim=True)

        #aff2 torch.Size([1, 1, 48, 48, 48])
        #aff torch.Size([1, 1, 48, 160, 160])

        b, c, h, w = imgs.shape
        _imgs = self.get_dilated_neighbors(imgs)
        _pos = self.pos.to(_imgs.device)

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1,1,_imgs.shape[self.dim],1,1)
        _pos_rep = _pos.repeat(b, 1, 1, h, w)

        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)
        _pos_std = torch.std(_pos_rep, dim=self.dim, keepdim=True)

        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1)**2
        aff = aff.mean(dim=1, keepdim=True)

        pos_aff = -(_pos_rep / (_pos_std + 1e-8) / self.w1)**2
        #pos_aff = pos_aff.mean(dim=1, keepdim=True)
        aff2 = F.interpolate(aff2.squeeze(0), size=imgs.size()[-2:], mode='bilinear', align_corners=True)
        aff = F.softmax(aff, dim=2) + F.softmax(aff2.unsqueeze(0), dim=2) + F.softmax(pos_aff, dim=2)

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks

class PAR1(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        _imgs = self.get_dilated_neighbors(imgs)
        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)
        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        aff = F.softmax(aff, dim=2)

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks

class PAR1a(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        _imgs = self.get_dilated_neighbors(imgs)
        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        aff = -(_imgs_abs  / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        aff = F.softmax(aff, dim=2)

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks

class PAR1b(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        _imgs = self.get_dilated_neighbors(imgs)
        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        aff = -(_imgs_abs ) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        aff = F.softmax(aff, dim=2)

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks




def tv_loss(input_t):
    temp1 = torch.cat((input_t[:, :, 1:, :], input_t[:, :, -1, :].unsqueeze(2)), 2)
    temp2 = torch.cat((input_t[:, :, :, 1:], input_t[:, :, :, -1].unsqueeze(3)), 3)
    temp = (input_t - temp1)**2 + (input_t - temp2)**2
    return temp.sum()


class PAR2(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        _imgs = self.get_dilated_neighbors(imgs)
        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)
        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        l = tv_loss(masks)
        aff = F.softmax(aff, dim=2)/l

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks

class PAR2a(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        _imgs = self.get_dilated_neighbors(imgs)
        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)
        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        l = tv_loss(masks)
        aff = F.softmax(aff, dim=2)

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = ((_masks * aff).sum(2))*l

        return masks

class PAR2b(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        _imgs = self.get_dilated_neighbors(imgs)
        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)
        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)


        temp1 = torch.cat((masks[:, :, 1:, :], masks[:, :, -1, :].unsqueeze(2)), 2)
        temp2 = torch.cat((masks[:, :, :, 1:], masks[:, :, :, -1].unsqueeze(3)), 3)
        temp = (masks - temp1) ** 2 + (masks - temp2) ** 2
        #print(temp.shape)#torch.Size([1, 3, 48, 160, 160])

        # aff torch.Size([1, 1, 48, 160, 160])
        aff = F.softmax(aff, dim=2)

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = ((_masks * aff).sum(2))/temp
            #print(masks.shape)
        return masks



import pandas as pd
class PAR3(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        #print(masks.shape)torch.Size([1, 2 or 3, 160, 160]
        _imgs = self.get_dilated_neighbors(imgs)
        #print(_imgs.shape)torch.Size([1, 3, 48, 160, 160])
        input_t = _imgs
        temp1 = torch.cat((input_t[:, :, :, 1:, :], input_t[:,:, :, -1, :].unsqueeze(3)), 3)
        temp2 = torch.cat((input_t[:, :, :, :, 1:], input_t[:,:, :, :, -1].unsqueeze(4)), 4)
        temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2
        #print(temp.shape)torch.Size([1, 3, 48, 160, 160])




        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)


        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        #temp = -(temp / (_imgs_std + 1e-8) ) ** 2
        temp = temp.mean(dim=1, keepdim=True)
        # aff torch.Size([1, 1, 48, 160, 160])
        aff = F.softmax(aff, dim=2) +  F.softmax(temp, dim=2)
        #print(aff.shape)torch.Size([1, 1, 48, 160, 160])

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks
class PAR3a(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        #print(masks.shape)torch.Size([1, 2 or 3, 160, 160]
        _imgs = self.get_dilated_neighbors(imgs)
        #print(_imgs.shape)torch.Size([1, 3, 48, 160, 160])
        input_t = _imgs
        temp1 = torch.cat((input_t[:, :, :, 1:, :], input_t[:,:, :, -1, :].unsqueeze(3)), 3)
        temp2 = torch.cat((input_t[:, :, :, :, 1:], input_t[:,:, :, :, -1].unsqueeze(4)), 4)
        temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2
        #print(temp.shape)torch.Size([1, 3, 48, 160, 160])

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)


        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        #temp = -(temp / (_imgs_std + 1e-8) ) ** 2
        temp = temp.mean(dim=1, keepdim=True)
        # aff torch.Size([1, 1, 48, 160, 160])
        aff = F.softmax(aff, dim=2) +  self.w2*F.softmax(temp, dim=2)
        #print(aff.shape)torch.Size([1, 1, 48, 160, 160])

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks

class PAR3b(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        #print(masks.shape)torch.Size([1, 2 or 3, 160, 160]
        _imgs = self.get_dilated_neighbors(imgs)
        #print(_imgs.shape)torch.Size([1, 3, 48, 160, 160])
        input_t = _imgs
        temp1 = torch.cat((input_t[:, :, :, 1:, :], input_t[:,:, :, -1, :].unsqueeze(3)), 3)
        temp2 = torch.cat((input_t[:, :, :, :, 1:], input_t[:,:, :, :, -1].unsqueeze(4)), 4)
        temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2
        #print(temp.shape)torch.Size([1, 3, 48, 160, 160])

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)


        aff = -((_imgs_abs / (_imgs_std + 1e-8) )* 5) ** 2
        #aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        #temp = -(temp / (_imgs_std + 1e-8) ) ** 2
        temp = temp.mean(dim=1, keepdim=True)
        # aff torch.Size([1, 1, 48, 160, 160])
        aff = F.softmax(aff, dim=2) - self.w2*F.softmax(temp, dim=2)
        #print(aff.shape)torch.Size([1, 1, 48, 160, 160])

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks

class PAR3b1(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):
        b, c, h, w = masks.shape
        l = tv_loss(masks) / h*w
        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        #print(masks.shape)torch.Size([1, 2 or 3, 160, 160]
        _imgs = self.get_dilated_neighbors(imgs)
        #print(_imgs.shape)torch.Size([1, 3, 48, 160, 160])
        input_t = _imgs
        temp1 = torch.cat((input_t[:, :, :, 1:, :], input_t[:,:, :, -1, :].unsqueeze(3)), 3)
        temp2 = torch.cat((input_t[:, :, :, :, 1:], input_t[:,:, :, :, -1].unsqueeze(4)), 4)
        temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2
        #print(temp.shape)torch.Size([1, 3, 48, 160, 160])

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)


        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        #temp = -(temp / (_imgs_std + 1e-8) ) ** 2
        temp = temp.mean(dim=1, keepdim=True)
        # aff torch.Size([1, 1, 48, 160, 160])
        aff = F.softmax(aff, dim=2) - self.w2*F.softmax(temp, dim=2)
        #print(aff.shape)torch.Size([1, 1, 48, 160, 160])

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)/ (l + 1)

        return masks

class PAR3b2(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):
        b, c, h, w = masks.shape
        l = tv_loss(masks) / h*w
        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        #print(masks.shape)torch.Size([1, 2 or 3, 160, 160]
        _imgs = self.get_dilated_neighbors(imgs)
        #print(_imgs.shape)torch.Size([1, 3, 48, 160, 160])
        input_t = _imgs
        temp1 = torch.cat((input_t[:, :, :, 1:, :], input_t[:,:, :, -1, :].unsqueeze(3)), 3)
        temp2 = torch.cat((input_t[:, :, :, :, 1:], input_t[:,:, :, :, -1].unsqueeze(4)), 4)
        temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2
        #print(temp.shape)torch.Size([1, 3, 48, 160, 160])

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)


        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        #temp = -(temp / (_imgs_std + 1e-8) ) ** 2
        temp = temp.mean(dim=1, keepdim=True)
        # aff torch.Size([1, 1, 48, 160, 160])
        aff = F.softmax(aff, dim=2) - self.w2*F.softmax(temp, dim=2)
        #print(aff.shape)torch.Size([1, 1, 48, 160, 160])

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)/ (l + 1e-8)

        return masks

class PAR3b3(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 4
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):
        b, c, h, w = masks.shape
        l = tv_loss(masks) / h*w
        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        #print(masks.shape)torch.Size([1, 2 or 3, 160, 160]
        _imgs = self.get_dilated_neighbors(imgs)
        #print(_imgs.shape)torch.Size([1, 3, 48, 160, 160])
        input_t = _imgs
        temp1 = torch.cat((input_t[:, :, :, 1:, :], input_t[:,:, :, -1, :].unsqueeze(3)), 3)
        temp2 = torch.cat((input_t[:, :, :, :, 1:], input_t[:,:, :, :, -1].unsqueeze(4)), 4)
        temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2
        #print(temp.shape)torch.Size([1, 3, 48, 160, 160])

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)


        aff = -(self.w1 * (_imgs_abs / (_imgs_std + 1e-8)) ) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        #temp = -(temp / (_imgs_std + 1e-8) ) ** 2
        temp = temp.mean(dim=1, keepdim=True)
        # aff torch.Size([1, 1, 48, 160, 160])
        aff = F.softmax(aff, dim=2) - self.w2*F.softmax(temp, dim=2)
        #print(aff.shape)torch.Size([1, 1, 48, 160, 160])

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)/ (l + 1e-8)

        return masks



class PAR3bb(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

        self.attn = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        #b, c, h, w = masks.shape

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        #print(masks.shape)torch.Size([1, 2 or 3, 160, 160]

        _imgs = self.get_dilated_neighbors(imgs)
        #print(_imgs.shape)torch.Size([1, 3, 48, 160, 160])
        input_t = _imgs
        temp1 = torch.cat((input_t[:, :, :, 1:, :], input_t[:,:, :, -1, :].unsqueeze(3)), 3)
        temp2 = torch.cat((input_t[:, :, :, :, 1:], input_t[:,:, :, :, -1].unsqueeze(4)), 4)
        temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2
        #print(temp.shape)torch.Size([1, 3, 48, 160, 160])

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)


        aff = -(_imgs_abs / (_imgs_std + 1e-8)/ self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        #temp = -(temp / (_imgs_std + 1e-8) ) ** 2
        temp = temp.mean(dim=1, keepdim=True)
        #print(self.kernel)
        #_x_pad = F.pad(temp, [1] * 4, mode='replicate', value=0)
        #_x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
        #temp = F.conv2d(_x_pad, self.kernel, dilation=1).view(b, c, -1, h, w)
        #print(temp.shape)
        # aff torch.Size([1, 1, 48, 160, 160])
        aff = (F.softmax(aff, dim=2) - self.w2*F.softmax(temp, dim=2)).clamp(0)
        #print(aff.shape)torch.Size([1, 1, 48, 160, 160])

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks




class PAR3c(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        #print(masks.shape)torch.Size([1, 2 or 3, 160, 160]
        _imgs = self.get_dilated_neighbors(imgs)
        #print(_imgs.shape)torch.Size([1, 3, 48, 160, 160])
        input_t = _imgs
        temp1 = torch.cat((input_t[:, :, :, 1:, :], input_t[:,:, :, -1, :].unsqueeze(3)), 3)
        temp2 = torch.cat((input_t[:, :, :, :, 1:], input_t[:,:, :, :, -1].unsqueeze(4)), 4)
        temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2
        #print(temp.shape)torch.Size([1, 3, 48, 160, 160])

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)

        l = tv_loss(masks)
        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        #temp = -(temp / (_imgs_std + 1e-8) ) ** 2
        temp = temp.mean(dim=1, keepdim=True)
        # aff torch.Size([1, 1, 48, 160, 160])
        aff = (F.softmax(aff, dim=2)) / (l + 1e-8) - self.w2*F.softmax(temp, dim=2)
        #print(aff.shape)torch.Size([1, 1, 48, 160, 160])

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks

class PAR3d(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        #print(masks.shape)torch.Size([1, 2 or 3, 160, 160]
        _imgs = self.get_dilated_neighbors(imgs)
        #print(_imgs.shape)torch.Size([1, 3, 48, 160, 160])
        input_t = _imgs
        temp1 = torch.cat((input_t[:, :, :, 1:, :], input_t[:,:, :, -1, :].unsqueeze(3)), 3)
        temp2 = torch.cat((input_t[:, :, :, :, 1:], input_t[:,:, :, :, -1].unsqueeze(4)), 4)
        temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2
        #print(temp.shape)torch.Size([1, 3, 48, 160, 160])

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)

        l = tv_loss(masks)
        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        #temp = -(temp / (_imgs_std + 1e-8) ) ** 2
        temp = temp.mean(dim=1, keepdim=True)
        # aff torch.Size([1, 1, 48, 160, 160])
        aff = F.softmax(aff, dim=2) - self.w2*F.softmax(temp, dim=2)
        #print(aff.shape)torch.Size([1, 1, 48, 160, 160])

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)/(l + 1e-8)

        return masks




class PAR3e(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        #print(masks.shape)torch.Size([1, 2 or 3, 160, 160]
        _imgs = self.get_dilated_neighbors(imgs)
        #print(_imgs.shape)torch.Size([1, 3, 48, 160, 160])
        input_t = _imgs
        temp1 = torch.cat((input_t[:, :, :, 1:, :], input_t[:,:, :, -1, :].unsqueeze(3)), 3)
        temp2 = torch.cat((input_t[:, :, :, :, 1:], input_t[:,:, :, :, -1].unsqueeze(4)), 4)
        temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2
        #print(temp.shape)torch.Size([1, 3, 48, 160, 160])

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)


        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        #temp = -(temp / (_imgs_std + 1e-8) ) ** 2
        temp = temp.mean(dim=1, keepdim=True)
        # aff torch.Size([1, 1, 48, 160, 160])
        aff = F.softmax(aff, dim=2) /  F.softmax(temp, dim=2)
        #aff = F.softmax(aff, dim=2) / self.w2 * F.softmax(temp, dim=2)  64.4
        #print(aff.shape)#torch.Size([1, 1, 48, 160, 160])

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks







class PAR3ey(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        #print(masks.shape)torch.Size([1, 2 or 3, 160, 160]
        _imgs = self.get_dilated_neighbors(imgs)
        #print(_imgs.shape)torch.Size([1, 3, 48, 160, 160])
        input_t = _imgs
        temp1 = torch.cat((input_t[:, :, :, 1:, :], input_t[:,:, :, -1, :].unsqueeze(3)), 3)
        temp2 = torch.cat((input_t[:, :, :, :, 1:], input_t[:,:, :, :, -1].unsqueeze(4)), 4)
        temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2
        #print(temp.shape)torch.Size([1, 3, 48, 160, 160])

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)

        imgs_1 = masks.view( -1)
        s = pd.Series(imgs_1.cpu().detach().numpy())
        #print(s.shape)
        Skewness = s.skew()

        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        #temp = -(temp / (_imgs_std + 1e-8) ) ** 2
        temp = temp.mean(dim=1, keepdim=True)
        # aff torch.Size([1, 1, 48, 160, 160])
        aff = F.softmax(aff, dim=2) -  self.w2*F.softmax(temp, dim=2)
        #print(aff.shape)torch.Size([1, 1, 48, 160, 160])

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)*np.abs(Skewness)

        return masks


class PAR3f(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        #print(masks.shape)torch.Size([1, 2 or 3, 160, 160]
        _imgs = self.get_dilated_neighbors(imgs)
        #print(_imgs.shape)torch.Size([1, 3, 48, 160, 160])
        input_t = _imgs
        temp1 = torch.cat((input_t[:, :, :, 1:, :], input_t[:,:, :, -1, :].unsqueeze(3)), 3)
        temp2 = torch.cat((input_t[:, :, :, :, 1:], input_t[:,:, :, :, -1].unsqueeze(4)), 4)
        temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2
        #print(temp.shape)torch.Size([1, 3, 48, 160, 160])

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)

        imgs_1 = masks.view( -1)
        s = pd.Series(imgs_1.cpu().detach().numpy())
        #print(s.shape)
        Skewness = s.skew()

        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        #temp = -(temp / (_imgs_std + 1e-8) ) ** 2
        temp = temp.mean(dim=1, keepdim=True)
        # aff torch.Size([1, 1, 48, 160, 160])
        aff = F.softmax(aff, dim=2) -  self.w2*F.softmax(temp, dim=2)
        #print(aff.shape)torch.Size([1, 1, 48, 160, 160])

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)*np.abs(Skewness)

        return masks

class PAR4(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        #print(masks.shape)torch.Size([1, 2 or 3, 160, 160]
        _imgs = self.get_dilated_neighbors(imgs)
        #print(_imgs.shape)torch.Size([1, 3, 48, 160, 160])
        input_t = _imgs
        temp1 = torch.cat((input_t[:, :, :, 1:, :], input_t[:,:, :, -1, :].unsqueeze(3)), 3)
        temp2 = torch.cat((input_t[:, :, :, :, 1:], input_t[:,:, :, :, -1].unsqueeze(4)), 4)
        temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2
        #print(temp.shape)torch.Size([1, 3, 48, 160, 160])

        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)
        imgs_1 = imgs.view( -1)
        s = pd.Series(imgs_1.cpu().detach().numpy())
        #print(s.shape)
        Skewness = s.skew()
        #s = pd.Series(i.cpu().detach().numpy())
        # kurt = s.kurt()
        # K.append(kurt)
        # K = np.vstack(K)
        # K = torch.from_numpy(K)
        # print(_imgs_abs.shape)
        # print(_imgs_std.shape)
        # torch.Size([1, 3, 48, 160, 160])
        # torch.Size([1, 3, 1, 160, 160])

        aff = -((temp * Skewness)/ (_imgs_std + 1e-8) / self.w1 ) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        # aff torch.Size([1, 1, 48, 160, 160])
        aff = F.softmax(aff, dim=2)
        #print(aff.shape)torch.Size([1, 1, 48, 160, 160])

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks
class PAR(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)

        b, c, h, w = imgs.shape
        _imgs = self.get_dilated_neighbors(imgs)
        _pos = self.pos.to(_imgs.device)

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _pos_rep = _pos.repeat(b, 1, 1, h, w)

        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)
        _pos_std = torch.std(_pos_rep, dim=self.dim, keepdim=True)

        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)

        pos_aff = -(_pos_rep / (_pos_std + 1e-8) / self.w1) ** 2
        # pos_aff = pos_aff.mean(dim=1, keepdim=True)

        aff = F.softmax(aff, dim=2) + self.w2 * F.softmax(pos_aff, dim=2)

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks