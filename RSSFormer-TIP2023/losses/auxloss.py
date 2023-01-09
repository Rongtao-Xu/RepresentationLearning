import torch
import torch.nn as nn
import torch.nn.functional as F




def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = F.cross_entropy(pred, label, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def soft_cross_entropy(pred,
                       label,
                       weight=None,
                       reduction='mean',
                       avg_factor=None):
    """Calculate the Soft CrossEntropy loss. The label can be float.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction with shape (N, C).
            When using "mixup", the label can be float.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = -label * F.log_softmax(pred, dim=-1)
    loss = loss.sum(dim=-1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):
    """Calculate the binary CrossEntropy loss with logits.

    Args:
        pred (torch.Tensor): The prediction with shape (N, *).
        label (torch.Tensor): The gt label with shape (N, *).
        weight (torch.Tensor, optional): Element-wise weight of loss with shape
             (N, ). Defaults to None.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' , loss
             is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert pred.dim() == label.dim()

    loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss

class CELoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(CELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        n_pred_ch, n_target_ch = cls_score.shape[1], label.shape[1]
        if n_pred_ch == n_target_ch:
            # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            label = torch.argmax(label, dim=1)
        else:
            label = torch.squeeze(label, dim=1)
        label = label.long()
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls



class CrossEntropyLoss(nn.Module):
    """Cross entropy loss.

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_soft (bool): Whether to use the soft version of CrossEntropyLoss.
            Defaults to False.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 sigmoid=False,
                 softmax=False,
                 reduction='mean',
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = sigmoid
        self.use_soft = softmax
        assert not (
                self.use_soft and self.use_sigmoid
        ), 'use_sigmoid and use_soft could not be set simultaneously'

        self.reduction = reduction
        self.loss_weight = loss_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_soft:
            self.cls_criterion = soft_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        n_pred_ch, n_target_ch = cls_score.shape[1], label.shape[1]
        if n_pred_ch == n_target_ch:
            label = torch.argmax(label, dim=1)
        else:
            label = torch.squeeze(label, dim=1)
        label = label.long()

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls

loss_consis = torch.nn.L1Loss()
class MCTransAuxLoss(CrossEntropyLoss):
    def __init__(self,**kwargs):
        super(MCTransAuxLoss, self).__init__(**kwargs)

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        # print('input cls_score',cls_score.shape)
        # print('input label',label.shape)
        # input
        # cls_score
        # torch.Size([8, 7])
        # input
        # label
        # torch.Size([8, 512, 512])

        # print('input label', label)
        #To one hot
        num_classes = cls_score.shape[1]
        one_hot = []
        for l in label:
            #print('input l', torch.unique(l).shape)
            one_hot.append(self.one_hot(torch.unique(l), num_classes=num_classes).sum(dim=0))
        label = torch.stack(one_hot)

        reduction = (
            reduction_override if reduction_override else self.reduction)
        #print('out cls_score',cls_score.shape)torch.Size([8, 7])
        #print('out cls_score',cls_score)
        #print('out label',label.shape)  torch.Size([8, 7])
        #print('out label',label)
        #loss_cls1 = loss_consis(cls_score, label)

        loss_cls1 = 1/(1 + torch.exp(abs(cls_score - label)))
        loss_cls1 = loss_cls1.sum(1)/(2*loss_cls1.shape[0])
        #print(loss_cls1.shape)
        #print(loss_cls1)
        #loss_cls1 = loss_consis(cls_score, label)
        loss_cls = 0
        # loss_cls = self.cls_criterion(
        #     cls_score,
        #     label,
        #     weight,
        #     reduction=reduction,
        #     avg_factor=avg_factor,
        #     **kwargs)
        #print('out loss_cls', loss_cls)
        return 0.5*loss_cls,loss_cls1#.cuda()#.softmax(dim=0)

    def one_hot(self, input, num_classes, dtype=torch.float):
        assert input.dim() > 0, "input should have dim of 1 or more."

        # if 1D, add singelton dim at the end
        if input.dim() == 1:
            input = input.view(-1, 1)

        sh = list(input.shape)

        assert sh[1] == 1, "labels should have a channel with length equals to one."
        sh[1] = num_classes

        o = torch.zeros(size=sh, dtype=dtype, device=input.device)
        labels = o.scatter_(dim=1, index=input.long(), value=1)

        return labels



class MCTransAuxLoss3(CrossEntropyLoss):
    def __init__(self,**kwargs):
        super(MCTransAuxLoss3, self).__init__(**kwargs)


    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        #print('input cls_score',cls_score.shape)
        #print('input label',label.shape)
        # print('input label', label)
        #To one hot
        num_classes = cls_score.shape[1]
        one_hot = []
        for l in label:
            #print('input l', torch.unique(l).shape)
            one_hot.append(self.one_hot(torch.unique(l), num_classes=num_classes).sum(dim=0))
        label = torch.stack(one_hot)


        # print('out cls_score',cls_score.shape)
        # # print('out cls_score',cls_score)
        # print('out label',label.shape)
        # print('out label',label)
        loss_cls1 = loss_consis(cls_score, label)
        loss_cls = softmax_focalloss(cls_score, label, gamma=loss_cls1)


        #print('out loss_cls', loss_cls)
        return loss_cls#.cuda()#.softmax(dim=0)

    def one_hot(self, input, num_classes, dtype=torch.float):
        assert input.dim() > 0, "input should have dim of 1 or more."

        # if 1D, add singelton dim at the end
        if input.dim() == 1:
            input = input.view(-1, 1)

        sh = list(input.shape)

        assert sh[1] == 1, "labels should have a channel with length equals to one."
        sh[1] = num_classes

        o = torch.zeros(size=sh, dtype=dtype, device=input.device)
        labels = o.scatter_(dim=1, index=input.long(), value=1)

        return labels

def softmax_focalloss(y_pred, y_true, ignore_index=-1, gamma=2.0, normalize=False):
    """
    Args:
        y_pred: [N, #class, H, W]
        y_true: [N, H, W] from 0 to #class
        gamma: scalar
    Returns:
    """

    y_pred = y_pred.unsqueeze(dim=1)
    y_true = y_true.unsqueeze(dim=1)
    print('y_pred',y_pred.shape)
    print('y_true',y_true.shape)
    losses = F.cross_entropy(y_pred, y_true, ignore_index=ignore_index, reduction='none')
    with torch.no_grad():
        p = y_pred.softmax(dim=1)
        modulating_factor = (1 - p).pow(gamma)
        valid_mask = ~ y_true.eq(ignore_index)
        masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
        print(masked_y_true.shape)#torch.Size([8, 7])
        print(masked_y_true.shape)
        # print(masked_y_true.reshpe(y_pred.shape[0],y_pred.shape[1],1,1).shape)
        # print(modulating_factor.reshpe(y_pred.shape[0],1,1,1).shape)

        # print(masked_y_true.unsqueeze(dim=2).unsqueeze(dim=3).shape)
        # print(modulating_factor.unsqueeze(dim=2).unsqueeze(dim=3).shape)
        modulating_factor = torch.gather(modulating_factor.unsqueeze(dim=1), dim=1, index=masked_y_true.to(torch.int64).unsqueeze(dim=1)).squeeze_(dim=1)
        scale = 1.
        if normalize:
            scale = losses.sum() / (losses * modulating_factor).sum()
    losses = scale * (losses * modulating_factor).sum() / (valid_mask.sum() + p.size(0))

    return losses



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def FocalLoss(input, target, gamma=0, alpha=None, size_average=True):

    gamma = gamma
    alpha = alpha
    size_average = size_average


    if input.dim()>2:
        input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
    target = target.view(-1,1)

    logpt = F.log_softmax(input)
    logpt = logpt.gather(1,target.to(torch.int64))
    logpt = logpt.view(-1)
    pt = Variable(logpt.data.exp())

    if alpha is not None:
        if alpha.type()!=input.data.type():
            alpha = alpha.type_as(input.data)
        at = alpha.gather(0,target.data.view(-1))
        logpt = logpt * Variable(at)

    loss = -1 * (1-pt)**gamma * logpt
    if size_average: return loss.mean()
    else: return loss.sum()