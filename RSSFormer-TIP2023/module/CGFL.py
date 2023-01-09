
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import ever as er
from losses.auxloss import MCTransAuxLoss3
from losses.auxloss import MCTransAuxLoss#3 as MCTransAuxLoss
from losses.obj2 import objLoss as DiscriminativeLoss
def _masked_ignore(y_pred: torch.Tensor, y_true: torch.Tensor, ignore_index: int = -1):
    # usually used for BCE-like losses
    y_pred = y_pred.reshape((-1,))
    y_true = y_true.reshape((-1,))
    valid = y_true != ignore_index
    y_true = y_true.masked_select(valid).float()
    y_pred = y_pred.masked_select(valid).float()
    return y_pred, y_true


def tversky_loss_with_logits(y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float, beta: float,
                             smooth_value: float = 1.0,
                             ignore_index: int = -1):
    y_pred, y_true = _masked_ignore(y_pred, y_true, ignore_index)

    y_pred = y_pred.sigmoid()
    tp = (y_pred * y_true).sum()
    # fp = (y_pred * (1 - y_true)).sum()
    fp = y_pred.sum() - tp
    # fn = ((1 - y_pred) * y_true).sum()
    fn = y_true.sum() - tp

    tversky_coeff = (tp + smooth_value) / (tp + alpha * fn + beta * fp + smooth_value)
    return 1. - tversky_coeff


def binary_cross_entropy_with_logits(output: torch.Tensor, target: torch.Tensor, reduction: str = 'mean',
                                     ignore_index: int = -1):
    output, target = _masked_ignore(output, target, ignore_index)
    return F.binary_cross_entropy_with_logits(output, target, reduction=reduction)


def softmax_focallossy(y_pred, y_true, ignore_index=-1, gamma=2.0, normalize=False):
    """
    Args:
        y_pred: [N, #class, H, W]
        y_true: [N, H, W] from 0 to #class
        gamma: scalar
    Returns:
    """
    # print(y_pred.shape)
    # print(y_true.shape)
    # torch.Size([8, 7, 512, 512])
    # torch.Size([8, 512, 512])
    losses = F.cross_entropy(y_pred, y_true, ignore_index=ignore_index, reduction='none')
    with torch.no_grad():
        p = y_pred.softmax(dim=1)
        modulating_factor = (1 - p).pow(gamma)
        valid_mask = ~ y_true.eq(ignore_index)
        masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
        # print(masked_y_true.unsqueeze(dim=1).shape)
        # print(modulating_factor.shape)
        # torch.Size([8, 1, 512, 512])
        # torch.Size([8, 7, 512, 512])
        modulating_factor = torch.gather(modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)).squeeze_(dim=1)
        scale = 1.
        if normalize:
            scale = losses.sum() / (losses * modulating_factor).sum()
    losses = scale * (losses * modulating_factor).sum() / (valid_mask.sum() + p.size(0))

    return losses

def softmax_focalloss(y_pred, y_true, ignore_index=-1, gamma=2, normalize=False):

    losses = F.cross_entropy(y_pred, y_true, ignore_index=ignore_index)
    with torch.no_grad():
        p = y_pred.softmax(dim=1)
        gamma = gamma

        # if gamma>= 5: 50.8
        #     gamma = 5
        # if gamma >= 3:
        #     gamma = 3
        #print(p.shape)torch.Size([8, 7, 512, 512])
        gamma = gamma.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        #print('gamma',gamma.shape)
        #print(losses)

        #modulating_factor = (1 - p)*1/(1+gamma)
        modulating_factor = (1 - p) *  (1 - gamma/7)
        valid_mask = ~ y_true.eq(ignore_index)
        masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
        # print(masked_y_true.unsqueeze(dim=1).shape)
        # print(modulating_factor.shape)
        # torch.Size([8, 1, 512, 512])
        # torch.Size([8, 7, 512, 512])
        modulating_factor = torch.gather(modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)).squeeze_(dim=1)
        scale = 1.

    losses = scale * (losses * modulating_factor).sum() / (valid_mask.sum() + p.size(0))

    return losses
import math
def cosine_annealing(lower_bound, upper_bound,gamma):
    gamma = gamma.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    return upper_bound*1/(1+gamma)

def FSG_softmax_focalloss(y_pred, y_true, ignore_index=255, gamma=2.0,
                                ):
    losses = F.cross_entropy(y_pred, y_true, ignore_index=ignore_index, reduction='none')
    with torch.no_grad():
        p = y_pred.softmax(dim=1)
        modulating_factor = (1 - p)
        valid_mask = ~ y_true.eq(ignore_index)
        masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
        modulating_factor = torch.gather(modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)).squeeze_(dim=1)
        normalizer = losses.sum() / (losses * modulating_factor).sum()
        scales = modulating_factor * normalizer
        scale = cosine_annealing(1, scales,gamma)

    losses = (losses * scale).sum() / (valid_mask.sum() + p.size(0))
    return losses



class SegmentationLoss(nn.Module):
    def __init__(self, loss_config):
        super(SegmentationLoss, self).__init__()
        self.loss_config = loss_config

        self.criterion_discriminative = DiscriminativeLoss(
            delta_var = 0.5, delta_dist = 2, norm=2)

        print('SegmentationLoss')
    def forward(self, y_pred, y_true: torch.Tensor):
        loss_dict = dict()
        #print('dis_loss')
        # mem = torch.cuda.max_memory_allocated() // 1024 // 1024
        # loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(y_pred.device)
        if 'ce' in self.loss_config:
            loss_ce = F.cross_entropy(y_pred, y_true.long(), ignore_index=-1)
            loss_dict['ce_loss'] = loss_ce
            # print('loss_ce',loss_ce)
            # print(y_pred.shape)
            # print(y_true.shape)
            # torch.Size([8, 7, 512, 512])
            # torch.Size([8, 512, 512])
            #y_predb = y_pred[:, 0, :, :]
            #print(y_predb.shape)
            #invalidmask = y_true == -1
            #bg_y_true = torch.where(y_true > 0, torch.ones_like(y_predb), torch.zeros_like(y_predb))
            #print(bg_y_true.shape)

            # loss_dict['ce_loss'] = self.criterion_aux(y_pred, y_true)
            # bg_y_true = y_pred
            # for i in range(7):
            #     bg_y_true[:, i, :, :] = torch.where(y_true == i, i*torch.ones_like(y_true), torch.zeros_like(y_true))
            # n_objects = y_pred.shape[0]
            # n_object = torch.IntTensor(n_objects)#n_objects*torch.ones([8])
            # # loss_dict['ce_loss'] = self.criterion_discriminative(y_pred.float(),bg_y_true.float(),n_object,20) + loss_ce


        if 'fcloss' in self.loss_config:
            loss_dict['fc_loss'] = softmax_focalloss(y_pred, y_true, gamma=self.loss_config.fcloss.gamma,
                                                     normalize=True)

        if 'bceloss' in self.loss_config:
            y_predb = y_pred[:, 0, :, :]
            invalidmask = y_true == -1
            bg_y_true = torch.where(y_true > 0, torch.ones_like(y_predb), torch.zeros_like(y_predb))
            bg_y_true[invalidmask] = -1
            loss_dict['bceloss'] = binary_cross_entropy_with_logits(y_predb, bg_y_true,
                                                                    ignore_index=-1) * self.loss_config.bceloss.scaler

        if 'tverloss' in self.loss_config:
            y_predb = y_pred[:, 0, :, :]
            invalidmask = y_true == -1
            bg_y_true = torch.where(y_true > 0, torch.ones_like(y_predb), torch.zeros_like(y_predb))
            bg_y_true[invalidmask] = -1
            loss_dict['tverloss'] = tversky_loss_with_logits(y_predb, bg_y_true, self.loss_config.tverloss.alpha,
                                                             self.loss_config.tverloss.beta,
                                                             ignore_index=-1) * self.loss_config.tverloss.scaler

        if 'diceloss' in self.loss_config:

            loss_dict['dice_loss'] = dice_loss_with_logits(y_pred, y_true)
            #loss_dict['ce_loss'] = F.cross_entropy(y_pred, y_true.long(), ignore_index=-1)


        return loss_dict


class SegmentationLossaux(nn.Module):
    def __init__(self, loss_config):
        super(SegmentationLossaux, self).__init__()
        self.loss_config = loss_config

        self.criterion_discriminative = DiscriminativeLoss(
            delta_var = 0.5, delta_dist = 2, norm=2)
        self.criterion_aux = MCTransAuxLoss()
        print('SegmentationLossaux')
    def forward(self, y_pred, y_true: torch.Tensor,y_pred2):
        loss_dict = dict()
        #print('dis_loss')
        # mem = torch.cuda.max_memory_allocated() // 1024 // 1024
        # loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(y_pred.device)
        if 'ce' in self.loss_config:
            #loss_ce = F.cross_entropy(y_pred, y_true.long(), ignore_index=-1)
            #print('loss_ce', loss_ce)

            y_predb = y_pred[:, 0, :, :]
            invalidmask = y_true == -1
            bg_y_true = torch.where(y_true > 0, torch.ones_like(y_predb), torch.zeros_like(y_predb))
            bg_y_true[invalidmask] = 0
            #y_pred, bg_y_true = _masked_ignore(y_predb, bg_y_true, ignore_index=-1)
            #loss_dict['ce_loss'] = loss_ce

            #loss_dict['aux3_loss'],l1= self.criterion_aux(y_pred2, bg_y_true)

            loss, l1 = self.criterion_aux(y_pred2, bg_y_true)
            #print('l1',l1)
            # similarity = torch.cosine_similarity(y_pred2, bg_y_true, dim=0)
            # similarity = torch.mean(similarity)
            # l_S = 1 - similarity
            result = l1 #round(float(l1), 1)
            #print(result)
            #loss_dict['ce_loss'] = abs(result)*loss_ce
            loss_dict['fc_loss'] = softmax_focalloss(y_pred, y_true, gamma=result)
            # print(y_pred.shape)
            # print(y_true.shape)
            # torch.Size([8, 7, 512, 512])
            # torch.Size([8, 512, 512])
            #y_predb = y_pred[:, 0, :, :]
            #print(y_predb.shape)
            #invalidmask = y_true == -1
            #bg_y_true = torch.where(y_true > 0, torch.ones_like(y_predb), torch.zeros_like(y_predb))
            #print(bg_y_true.shape)
            # bg_y_true = y_pred
            # for i in range(7):
            #     bg_y_true[:, i, :, :] = torch.where(y_true == i, i*torch.ones_like(y_true), torch.zeros_like(y_true))
            # loss_consis = torch.nn.L1Loss()
            # l1 = loss_consis(y_pred2, bg_y_true)

            #result = round(float(l1), 1)


        if 'fcloss' in self.loss_config:
            loss_dict['fc_loss'] = softmax_focalloss(y_pred, y_true, gamma=self.loss_config.fcloss.gamma,
                                                     normalize=True)

        if 'bceloss' in self.loss_config:
            y_predb = y_pred[:, 0, :, :]
            invalidmask = y_true == -1
            bg_y_true = torch.where(y_true > 0, torch.ones_like(y_predb), torch.zeros_like(y_predb))
            bg_y_true[invalidmask] = -1
            loss_dict['bceloss'] = binary_cross_entropy_with_logits(y_predb, bg_y_true,
                                                                    ignore_index=-1) * self.loss_config.bceloss.scaler

        if 'tverloss' in self.loss_config:
            y_predb = y_pred[:, 0, :, :]
            invalidmask = y_true == -1
            bg_y_true = torch.where(y_true > 0, torch.ones_like(y_predb), torch.zeros_like(y_predb))
            bg_y_true[invalidmask] = -1
            loss_dict['tverloss'] = tversky_loss_with_logits(y_predb, bg_y_true, self.loss_config.tverloss.alpha,
                                                             self.loss_config.tverloss.beta,
                                                             ignore_index=-1) * self.loss_config.tverloss.scaler

        if 'diceloss' in self.loss_config:

            loss_dict['dice_loss'] = dice_loss_with_logits(y_pred, y_true) * self.loss_config.diceloss.scaler
            #loss_dict['ce_loss'] = F.cross_entropy(y_pred, y_true.long(), ignore_index=-1)


        return loss_dict

class SegmentationLossauxy(nn.Module):
    def __init__(self, loss_config):
        super(SegmentationLossauxy, self).__init__()
        self.loss_config = loss_config

        self.criterion_discriminative = DiscriminativeLoss(
            delta_var = 0.5, delta_dist = 2, norm=2)
        self.criterion_aux = MCTransAuxLoss()
        print('SegmentationLossaux')
    def forward(self, y_pred, y_true: torch.Tensor,y_pred2):
        loss_dict = dict()
        #print('dis_loss')
        # mem = torch.cuda.max_memory_allocated() // 1024 // 1024
        # loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(y_pred.device)
        if 'ce' in self.loss_config:
            loss_ce = F.cross_entropy(y_pred, y_true.long(), ignore_index=-1)
            #print('loss_ce', loss_ce)

            y_predb = y_pred[:, 0, :, :]
            invalidmask = y_true == -1
            bg_y_true = torch.where(y_true > 0, torch.ones_like(y_predb), torch.zeros_like(y_predb))
            bg_y_true[invalidmask] = 0
            #y_pred, bg_y_true = _masked_ignore(y_predb, bg_y_true, ignore_index=-1)


            #loss_dict['aux3_loss'],l1= self.criterion_aux(y_pred2, bg_y_true)
            loss, l1 = self.criterion_aux(y_pred2, bg_y_true)
            loss_dict['ce_loss'] = abs(l1)*loss_ce
            #loss_dict['fc_loss'] = l1
            # print(y_pred.shape)
            # print(y_true.shape)
            # torch.Size([8, 7, 512, 512])
            # torch.Size([8, 512, 512])
            #y_predb = y_pred[:, 0, :, :]
            #print(y_predb.shape)
            #invalidmask = y_true == -1
            #bg_y_true = torch.where(y_true > 0, torch.ones_like(y_predb), torch.zeros_like(y_predb))
            #print(bg_y_true.shape)
            # bg_y_true = y_pred
            # for i in range(7):
            #     bg_y_true[:, i, :, :] = torch.where(y_true == i, i*torch.ones_like(y_true), torch.zeros_like(y_true))



            # loss_consis = torch.nn.L1Loss()
            # l1 = loss_consis(y_pred2, bg_y_true)

            result = round(float(l1), 1)
            loss_dict['fc_loss'] = softmax_focalloss(y_pred, y_true, gamma=result,
                                                      normalize=True)
            # loss_dict['fc_loss'] = softmax_focalloss(y_pred, y_true, gamma=2.0,
            #                                           normalize=True)
        if 'fcloss' in self.loss_config:
            loss_dict['fc_loss'] = softmax_focalloss(y_pred, y_true, gamma=self.loss_config.fcloss.gamma,
                                                     normalize=True)

        if 'bceloss' in self.loss_config:
            y_predb = y_pred[:, 0, :, :]
            invalidmask = y_true == -1
            bg_y_true = torch.where(y_true > 0, torch.ones_like(y_predb), torch.zeros_like(y_predb))
            bg_y_true[invalidmask] = -1
            loss_dict['bceloss'] = binary_cross_entropy_with_logits(y_predb, bg_y_true,
                                                                    ignore_index=-1) * self.loss_config.bceloss.scaler

        if 'tverloss' in self.loss_config:
            y_predb = y_pred[:, 0, :, :]
            invalidmask = y_true == -1
            bg_y_true = torch.where(y_true > 0, torch.ones_like(y_predb), torch.zeros_like(y_predb))
            bg_y_true[invalidmask] = -1
            loss_dict['tverloss'] = tversky_loss_with_logits(y_predb, bg_y_true, self.loss_config.tverloss.alpha,
                                                             self.loss_config.tverloss.beta,
                                                             ignore_index=-1) * self.loss_config.tverloss.scaler

        if 'diceloss' in self.loss_config:

            loss_dict['dice_loss'] = dice_loss_with_logits(y_pred, y_true) * self.loss_config.diceloss.scaler
            #loss_dict['ce_loss'] = F.cross_entropy(y_pred, y_true.long(), ignore_index=-1)


        return loss_dict




def dice_coeff(y_pred, y_true, weights: torch.Tensor, smooth_value: float = 1.0, ):
    y_pred = y_pred[:, weights]
    y_true = y_true[:, weights]
    inter = torch.sum(y_pred * y_true, dim=0)
    z = y_pred.sum(dim=0) + y_true.sum(dim=0) + smooth_value

    return ((2 * inter + smooth_value) / z).mean()


def select(y_pred: torch.Tensor, y_true: torch.Tensor, ignore_index: int):
    assert y_pred.ndim == 4 and y_true.ndim == 3
    c = y_pred.size(1)
    y_pred = y_pred.permute(0, 2, 3, 1).reshape(-1, c)
    y_true = y_true.reshape(-1)

    valid = y_true != ignore_index

    y_pred = y_pred[valid, :]
    y_true = y_true[valid]
    return y_pred, y_true


def dice_loss_with_logits(y_pred: torch.Tensor, y_true: torch.Tensor,
                          smooth_value: float = 1.0,
                          ignore_index: int = -1,
                          ignore_channel: int = -1):
    c = y_pred.size(1)
    y_pred, y_true = select(y_pred, y_true, ignore_index)
    weight = torch.as_tensor([True] * c, device=y_pred.device)
    if c == 1:
        y_prob = y_pred.sigmoid()
        return 1. - dice_coeff(y_prob, y_true.reshape(-1, 1), weight, smooth_value)
    else:
        y_prob = y_pred.softmax(dim=1)
        y_true = F.one_hot(y_true, num_classes=c)
        if ignore_channel != -1:
            weight[ignore_channel] = False

        return 1. - dice_coeff(y_prob, y_true, weight, smooth_value)


def label_smoothing_dice_loss_with_logits(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 0.1,
                                          smooth_value: float = 1.0,
                                          ignore_index: int = -1,
                                          ignore_channel: int = -1,
                                          ):
    y_true = torch.where(y_true == 0, y_true + eps, y_true - eps)
    c = y_pred.size(1)
    y_pred, y_true = select(y_pred, y_true, ignore_index)
    weight = torch.as_tensor([True] * c, device=y_pred.device)
    if c == 1:
        y_prob = y_pred.sigmoid()
        return 1. - dice_coeff(y_prob, y_true.reshape(-1, 1), weight, smooth_value)
    else:
        y_prob = y_pred.softmax(dim=1)
        y_true = F.one_hot(y_true, num_classes=c)
        if ignore_channel != -1:
            weight[ignore_channel] = False

        return 1. - dice_coeff(y_prob, y_true, weight, smooth_value)


def multi_binary_label(batched_mask: torch.Tensor, num_classes):
    labels = []
    for cls in range(1, num_classes + 1):
        labels.append((batched_mask == cls).to(torch.long))
    return labels


def label_smoothing_binary_cross_entropy(output: torch.Tensor, target: torch.Tensor, eps: float = 0.1,
                                         reduction: str = 'mean', ignore_index: int = -1):
    output, target = _masked_ignore(output, target, ignore_index)
    target = torch.where(target == 0, target + eps, target - eps)
    return F.binary_cross_entropy_with_logits(output, target, reduction=reduction)


def multi_binary_loss(y_pred, y_true, num_classes, reduction='mean', label_smooth=0., dice_scaler=0., bce_scaler=0.):
    labels = multi_binary_label(y_true, num_classes)
    losses = []
    for cls in range(num_classes):
        bipred = y_pred[:, cls, :, :]
        bipred = bipred.reshape(y_pred.size(0), 1, y_pred.size(2), y_pred.size(3)).contiguous()
        if label_smooth > 0:
            bce_loss = label_smoothing_binary_cross_entropy(bipred, labels[cls].reshape_as(bipred).float(),
                                                            label_smooth)
            dice_loss = label_smoothing_dice_loss_with_logits(bipred, labels[cls], label_smooth)
        else:
            bce_loss = binary_cross_entropy_with_logits(bipred, labels[cls].reshape_as(bipred).float(), ignore_index=-1)
            dice_loss = dice_loss_with_logits(bipred, labels[cls], ignore_index=-1)
        losses.append(bce_loss * bce_scaler + dice_loss * dice_scaler)

    if 'sum' == reduction:
        tloss = sum(losses)
    elif 'mean' == reduction:
        tloss = sum(losses) / float(num_classes)
    else:
        raise ValueError()

    if label_smooth > 0:
        return dict(mb_sm_loss=tloss)
    else:
        return dict(mb_loss=tloss)


