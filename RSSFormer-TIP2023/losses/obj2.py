from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import numpy as np

from torch import nn as nn


def calculate_variance_term(pred, gt, n_objects, delta_v, norm=2):
    """pred: bs, height * width, n_filters
       gt: bs, height * width, n_instances
       means: bs, n_instances, n_filters"""

    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)

    # bs, n_loc, n_instances, n_filters
    #means = means.unsqueeze(1).expand(bs, n_loc, n_instances, n_filters)
    # bs, n_loc, n_instances, n_filters
    pred = pred.unsqueeze(2).expand(bs, n_loc, n_instances, n_filters)
    # bs, n_loc, n_instances, n_filters
    gt = gt.unsqueeze(3).expand(bs, n_loc, n_instances, n_filters)

    # _var = (torch.clamp(torch.norm((pred - means), norm, 3) -
    #                     delta_v, min=0.0) ** 2) * gt[:, :, :, 0]
    _var = (torch.clamp(torch.norm((pred - gt), norm, 3) , min=0.0) ) * gt[:, :, :, 0]
    _var = torch.sum(_var)
    print('_var',_var.shape)
    var_term = 0.0
    # for i in range(bs):
    #     _var_sample = _var[i, :, :n_objects[i]]  # n_loc, n_objects
    #     _gt_sample = gt[i, :, :n_objects[i], 0]  # n_loc, n_objects
    #
    #     var_term += torch.sum(_var_sample) / torch.sum(_gt_sample)
    var_term = _var #/ bs

    return var_term


def calculate_distance_term(means, n_objects, delta_d, norm=2, usegpu=True):
    """means: bs, n_instances, n_filters"""

    bs, n_instances, n_filters = means.size()

    dist_term = 0.0
    for i in range(bs):
        _n_objects_sample = int(n_objects[i])

        if _n_objects_sample <= 1:
            continue

        _mean_sample = means[i, : _n_objects_sample, :]  # n_objects, n_filters
        means_1 = _mean_sample.unsqueeze(1).expand(
            _n_objects_sample, _n_objects_sample, n_filters)
        means_2 = means_1.permute(1, 0, 2)

        diff = means_1 - means_2  # n_objects, n_objects, n_filters

        _norm = torch.norm(diff, norm, 2)

        margin = 2 * delta_d * (1.0 - torch.eye(_n_objects_sample))
        if usegpu:
            margin = margin.cuda()
        margin = Variable(margin)

        _dist_term_sample = torch.sum(
            torch.clamp(margin - _norm, min=0.0) ** 2)
        _dist_term_sample = _dist_term_sample / \
            (_n_objects_sample * (_n_objects_sample - 1))
        dist_term += _dist_term_sample

    dist_term = dist_term / bs

    return dist_term


def calculate_regularization_term(means, n_objects, norm):
    """means: bs, n_instances, n_filters"""

    bs, n_instances, n_filters = means.size()

    reg_term = 0.0
    for i in range(bs):
        _mean_sample = means[i, : n_objects[i], :]  # n_objects, n_filters
        _norm = torch.norm(_mean_sample, norm, 1)
        reg_term += torch.mean(_norm)
    reg_term = reg_term / bs

    return reg_term

import torch.nn.functional as f
def _compute_unlabeled_push(cluster_means, pred, gt ,norm,delta_dist):


    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)

    # bs, n_loc, n_instances, n_filters
    cluster_means = cluster_means.unsqueeze(1).expand(bs, n_loc, n_instances, n_filters)
    # bs, n_loc, n_instances, n_filters
    embeddings = pred.unsqueeze(2).expand(bs, n_loc, n_instances, n_filters)
    #print(n_instances)
    #print(embeddings.shape)(2, 262144, 16)
    #print(cluster_means.shape)(2, 20, 16)

    # decrease number of instances `C` since we're ignoring 0-label
    n_instances -= 1
    # if there is only 0-label in the target return 0
    if n_instances == 0:
        return 0.

    background_mask = gt == 0
    n_background = background_mask.sum()
    background_push = 0.
    # skip embedding corresponding to the background pixels
    for cluster_mean in cluster_means[1:]:
        # compute distances between embeddings and a given cluster_mean
        dist_to_mean = torch.norm(embeddings - cluster_mean, norm, dim=-1)
        # apply background mask and compute hinge
        dist_hinged = torch.clamp((delta_dist - dist_to_mean) * background_mask.type(torch.cuda.FloatTensor), min=0) ** 2
        background_push += torch.sum(dist_hinged) / n_background

    # normalize by the number of instances
    return background_push / n_instances
def _compute_unlabeled_push2(var_term,cluster_means, pred, gt ,norm,delta_unlabeled):


    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)

    # bs, n_loc, n_instances, n_filters
    cluster_means = cluster_means.unsqueeze(1).expand(bs, n_loc, n_instances, n_filters)
    # bs, n_loc, n_instances, n_filters
    embeddings = pred.unsqueeze(2).expand(bs, n_loc, n_instances, n_filters)

    # decrease number of instances `C` since we're ignoring 0-label
    n_instances -= 1
    # if there is only 0-label in the target return 0
    if n_instances == 0:
        return 0.

    background_mask = gt == 0
    n_background = background_mask.sum()
    background_push = 0.
    # skip embedding corresponding to the background pixels
    for cluster_mean in cluster_means[1:]:
        # compute distances between embeddings and a given cluster_mean
        dist_to_mean = torch.norm(embeddings - cluster_mean, norm, dim=-1)
        # apply background mask and compute hinge
        dist_hinged = torch.clamp((delta_unlabeled * 2 - dist_to_mean) * background_mask.type(torch.cuda.FloatTensor), min=0)# ** 2
        background_push += torch.sum(dist_hinged) / n_background

    # normalize by the number of instances
    return (1*(background_push / n_instances) + 3*var_term)** 2
def discriminative_loss(input, target, n_objects,
                        max_n_objects, delta_v, delta_d, norm, usegpu=True):
    """input: bs, n_filters, fmap, fmap
       target: bs, n_instances, fmap, fmap
       n_objects: bs"""

    # print(input.shape)
    # print(target.shape)
    # torch.Size([8, 7, 512, 512])
    # torch.Size([8, 512, 512])

    delta_unlabeled = 1.5
    bs, n_filters, height, width = input.size()
    n_instances = target.size(1)

    input = input.permute(0, 2, 3, 1).contiguous().view(
        bs, height * width, n_filters)
    target = target.permute(0, 2, 3, 1).contiguous().view(
        bs, height * width, n_instances)


    var_term = calculate_variance_term(
        input, target, n_objects, delta_v, norm)

    unlabeled_push_weight = 1.0

    #unlabeled_push_term = _compute_unlabeled_push2(var_term,cluster_means, input, target,norm,delta_unlabeled)
    loss = unlabeled_push_weight*var_term#* unlabeled_push_term
    print('dis_loss',loss)
    return loss


def AbstractContrastiveLoss(input_, target):
    """
    Abstract class for all contrastive-based losses.
    This implementation expands all tensors to match the instance dimensions.
    This means that it's fast, but has high memory footprint.
    """
    def _compute_variance_term(self, cluster_means, embeddings, target, instance_counts, ignore_zero_label):
        """
        Computes the variance term, i.e. intra-cluster pull force that draws embeddings towards the mean embedding

        C - number of clusters (instances)
        E - embedding dimension
        SPATIAL - volume shape, i.e. DxHxW for 3D/ HxW for 2D

        Args:
            cluster_means: mean embedding of each instance, tensor (CxE)
            embeddings: embeddings vectors per instance, tensor (ExSPATIAL)
            target: label tensor (1xSPATIAL); each label is represented as one-hot vector
            instance_counts: number of voxels per instance
            ignore_zero_label: if True ignores the cluster corresponding to the 0-label
        """

        assert target.dim() in (2, 3)
        n_instances = cluster_means.shape[0]

        # compute the spatial mean and instance fields by scattering with the
        # target tensor
        cluster_means_spatial = cluster_means[target]
        instance_sizes_spatial = instance_counts[target]

        # permute the embedding dimension to axis 0
        if target.dim() == 2:
            cluster_means_spatial = cluster_means_spatial.permute(2, 0, 1)
        else:
            cluster_means_spatial = cluster_means_spatial.permute(3, 0, 1, 2)

        # compute the distance to cluster means
        dist_to_mean = torch.norm(embeddings - cluster_means_spatial, self.norm, dim=0)

        if ignore_zero_label:
            # zero out distances corresponding to 0-label cluster, so that it does not contribute to the losses
            dist_mask = torch.ones_like(dist_to_mean)
            dist_mask[target == 0] = 0
            dist_to_mean = dist_to_mean * dist_mask
            # decrease number of instances
            n_instances -= 1
            # if there is only 0-label in the target return 0
            if n_instances == 0:
                return 0.

        # zero out distances less than delta_var (hinge)
        hinge_dist = torch.clamp(dist_to_mean - self.delta_var, min=0) ** 2

        # normalize the variance by instance sizes and number of instances and sum it up
        variance_term = torch.sum(hinge_dist / instance_sizes_spatial) / n_instances
        return variance_term

    def _compute_unlabeled_push(self, cluster_means, embeddings, target):
        assert target.dim() in (2, 3)
        n_instances = cluster_means.shape[0]

        # permute embedding dimension at the end
        if target.dim() == 2:
            embeddings = embeddings.permute(1, 2, 0)
        else:
            embeddings = embeddings.permute(1, 2, 3, 0)

        # decrease number of instances `C` since we're ignoring 0-label
        n_instances -= 1
        # if there is only 0-label in the target return 0
        if n_instances == 0:
            return 0.

        background_mask = target == 0
        n_background = background_mask.sum()
        background_push = 0.
        # skip embedding corresponding to the background pixels
        for cluster_mean in cluster_means[1:]:
            # compute distances between embeddings and a given cluster_mean
            dist_to_mean = torch.norm(embeddings - cluster_mean, self.norm, dim=-1)
            # apply background mask and compute hinge
            dist_hinged = torch.clamp((self.delta_dist - dist_to_mean) * background_mask, min=0) ** 2
            background_push += torch.sum(dist_hinged) / n_background

        # normalize by the number of instances
        return background_push / n_instances

    def _compute_distance_term(self, cluster_means, ignore_zero_label):
        """
        Compute the distance term, i.e an inter-cluster push-force that pushes clusters away from each other, increasing
        the distance between cluster centers

        Args:
            cluster_means: mean embedding of each instance, tensor (CxE)
            ignore_zero_label: if True ignores the cluster corresponding to the 0-label
        """
        C = cluster_means.size(0)
        if C == 1:
            # just one cluster in the batch, so distance term does not contribute to the losses
            return 0.

        # expand cluster_means tensor in order to compute the pair-wise distance between cluster means
        # CxE -> CxCxE
        cluster_means = cluster_means.unsqueeze(0)
        shape = list(cluster_means.size())
        shape[0] = C

        # cm_matrix1 is CxCxE
        cm_matrix1 = cluster_means.expand(shape)
        # transpose the cluster_means matrix in order to compute pair-wise distances
        cm_matrix2 = cm_matrix1.permute(1, 0, 2)
        # compute pair-wise distances between cluster means, result is a CxC tensor
        dist_matrix = torch.norm(cm_matrix1 - cm_matrix2, p=self.norm, dim=2)

        # create matrix for the repulsion distance (i.e. cluster centers further apart than 2 * delta_dist
        # are not longer repulsed)
        repulsion_dist = 2 * self.delta_dist * (1 - torch.eye(C))
        repulsion_dist = repulsion_dist.to(cluster_means.device)

        if ignore_zero_label:
            if C == 2:
                # just two cluster instances, including one which is ignored, i.e. distance term does not contribute to the losses
                return 0.
            # set the distance to 0-label to be greater than 2*delta_dist, so that it does not contribute to the losses because of the hinge at 2*delta_dist

            # find minimum dist
            d_min = torch.min(dist_matrix[dist_matrix > 0]).item()
            # dist_multiplier = 2 * delta_dist / d_min + epsilon
            dist_multiplier = 2 * self.delta_dist / d_min + 1e-3
            # create distance mask
            dist_mask = torch.ones_like(dist_matrix)
            dist_mask[0, 1:] = dist_multiplier
            dist_mask[1:, 0] = dist_multiplier

            # mask the dist_matrix
            dist_matrix = dist_matrix * dist_mask
            # decrease number of instances
            C -= 1

        # zero out distances grater than 2*delta_dist (hinge)
        hinged_dist = torch.clamp(repulsion_dist - dist_matrix, min=0) ** 2
        # sum all of the hinged pair-wise distances
        dist_sum = torch.sum(hinged_dist)
        # normalized by the number of paris and return
        distance_term = dist_sum / (C * (C - 1))
        return distance_term

    def _compute_regularizer_term(self, cluster_means):
        """
        Computes the regularizer term, i.e. a small pull-force that draws all clusters towards origin to keep
        the network activations bounded
        """
        # compute the norm of the mean embeddings
        norms = torch.norm(cluster_means, p=self.norm, dim=1)
        # return the average norm per batch
        return torch.sum(norms) / cluster_means.size(0)

    def compute_instance_term(self, embeddings, cluster_means, target):
        """
        Computes auxiliary losses based on embeddings and a given list of target instances together with their mean embeddings

        Args:
            embeddings (torch.tensor): pixel embeddings (ExSPATIAL)
            cluster_means (torch.tensor): mean embeddings per instance (CxExSINGLETON_SPATIAL)
            target (torch.tensor): ground truth instance segmentation (SPATIAL)

        Returns:
            float: value of the instance-based term
        """
        raise NotImplementedError



    delta_var=0.5
    delta_dist=2.0
    norm='fro'
    alpha=1.
    beta=1.
    gamma=0.001
    unlabeled_push_weight=0.0,
    instance_term_weight=1.0

    delta_var = delta_var
    delta_dist = delta_dist
    norm = norm
    alpha = alpha
    beta = beta
    gamma = gamma
    unlabeled_push_weight = unlabeled_push_weight
    unlabeled_push = unlabeled_push_weight > 0
    instance_term_weight = instance_term_weight

    n_batches = input_.shape[0]
    # compute the losses per each instance in the batch separately
    # and sum it up in the per_instance variable
    loss = 0.
    for single_input, single_target in zip(input_, target):
        contains_bg = 0 in single_target
        if unlabeled_push and contains_bg:
            ignore_zero_label = True

        # get number of instances in the batch instance
        instance_ids, instance_counts = torch.unique(single_target, return_counts=True)

        # get the number of instances
        C = instance_ids.size(0)

        # compare spatial dimensions
        assert single_input.size()[1:] == single_target.size()

        # compute mean embeddings (output is of shape CxE)
        cluster_means = calculate_means(single_input, single_target, C)

        # compute variance term, i.e. pull force
        variance_term = _compute_variance_term(cluster_means, single_input, single_target, instance_counts,
                                                    ignore_zero_label)

        # compute unlabeled push force, i.e. push force between the mean cluster embeddings and embeddings of background pixels
        # compute only ignore_zero_label is True, i.e. a given patch contains background label
        unlabeled_push_term = 0.
        if unlabeled_push and contains_bg:
            unlabeled_push_term = _compute_unlabeled_push(cluster_means, single_input, single_target)

        # compute the instance-based auxiliary losses
        instance_term = compute_instance_term(single_input, cluster_means, single_target)

        # compute distance term, i.e. push force
        distance_term = _compute_distance_term(cluster_means, ignore_zero_label)

        # compute regularization term
        regularization_term = _compute_regularizer_term(cluster_means)

        # compute total losses and sum it up
        loss = alpha * variance_term + \
               beta * distance_term + \
               gamma * regularization_term + \
               instance_term_weight * instance_term + \
               unlabeled_push_weight * unlabeled_push_term

        loss += loss

    # reduce across the batch dimension
    return loss.div(n_batches)



class objLoss(_Loss):

    def __init__(self, delta_var, delta_dist, norm,
                 size_average=True, reduce=True, usegpu=True):
        super(objLoss, self).__init__(size_average)
        self.reduce = reduce

        #assert self.size_average
        #assert self.reduce

        self.delta_var = float(delta_var)
        self.delta_dist = float(delta_dist)
        self.norm = int(norm)
        self.usegpu = usegpu

        assert self.norm in [1, 2]

    def forward(self, input, target, n_objects, max_n_objects):
        #_assert_no_grad(target)
        print('-------------------------------------')
        print('-------------------------------------')
        return discriminative_loss(input, target, n_objects, max_n_objects,
                                   self.delta_var, self.delta_dist, self.norm,
                                   self.usegpu)
