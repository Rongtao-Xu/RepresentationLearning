#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:CASIA
# Time  :2022/4/9 7:23
import  argparse
from util.util import *
import random
import numpy as np
import torch
import os
from dataset import dataset as datautils
from model import model_dcl as modelutils
from easydict import EasyDict as edict
from util.Dice_test import getDice,getIoU
from util import html
import cv2
def accuracy(pred_mask, label):
    '''
    acc=(TP+TN)/(TP+FN+TN+FP)
    '''
    pred_mask = pred_mask.astype(np.uint8)
    TP, FN, TN, FP = [0, 0, 0, 0]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                if pred_mask[i][j] == 1:
                    TP += 1
                elif pred_mask[i][j] == 0:
                    FN += 1
            elif label[i][j] == 0:
                if pred_mask[i][j] == 1:
                    FP += 1
                elif pred_mask[i][j] == 0:
                    TN += 1
    acc = (TP + TN) / (TP + FN + TN + FP)
    sen = TP / (TP + FN)
    pre=TP/(TP+FP)
    return acc, sen,pre
parser = argparse.ArgumentParser()
parser.add_argument('-config', default='config.yml', help='config file')
parser.add_argument('-name', type=str,required=True,help='name of the model.')
parser.add_argument('-gpu_ids', type=str, default='0', help='gpu ids: 0 / 0,1,2')
parser.add_argument('-th', type=int, default=140, help='thresh')

config_file = edict(vars(parser.parse_args()))
# Reading configuration
a = cfg_from_file(parser.parse_args().config)
# Merge configuration
merge_a_into_b(config_file,a)
# set gpu ids
str_ids = a.gpu_ids.split(',')
a.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        a.gpu_ids.append(id)
if len(a.gpu_ids) > 0:
    torch.cuda.set_device(a.gpu_ids[0])
a.isTrain = False

def main():
    # set random seed
    seed=random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    np.random.seed(seed)
    random.seed(seed)
    for k, v in a.items():
        print(k, "=", v)
    isTrain=False
    # create a dataset given opt.dataset_mode and other options
    dataset = datautils.datasetcreater(a,isTrain)
    # create a model given opt.model and other options
    model = modelutils.modelcreater(a,isTrain)
    # regular setup: load and print networks; create schedulers
    model.setup(a)
    # create web dir to save resualt
    web_dir = os.path.join(a.results_dir, a.name, '%s_%s' % (a.name, a.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (a.name, a.phase, a.epoch))
    if a.eval:
        model.eval()
    Acc=0
    Dice=0
    Iou=0
    Sen=0
    Pre=0
    for i, data in enumerate(dataset):
        # unpack data from data loader
        model.set_input(data)
        predict, groundtruth = model.test2()
        predict = tensor2im(predict)
        groundtruth = tensor2im(groundtruth)
        predi_mask = np.zeros(shape=np.shape(predict))
        predi_mask[predict > a.th] = 1
        gt = np.zeros(shape=np.shape(groundtruth))
        gt[groundtruth > a.th] = 1

        #rw, gt = cv2.threshold(groundtruth, 150, 255, cv2.THRESH_BINARY)
        #rw, pr = cv2.threshold(predict, 150, 255, cv2.THRESH_BINARY)
        #acc, sen, pre = accuracy(pr, gt)

        acc,sen,pre=accuracy(predi_mask[:, :, 0],gt[:, :, 0])
        #print(acc,sen,pre)
        Pre+=pre
        Acc+=acc
        Sen += sen
        dice = getDice(groundtruth, predict,a.th)
        Dice += dice
        Iou+= getIoU(groundtruth, predict,a.th)
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        #print(dice)
        #opt.data_path

        if i % 100 == 0:
            print('Testing (%d/%d)' % (i,dataset.__len__()))
        #save_images(webpage, visuals, img_path, aspect_ratio=a.aspect_ratio, width=a.display_winsize)

    # save the HTML
    webpage.save()
    print("Mean Dice:" + str(Dice / len(dataset)))
    print("Mean IOU:" + str(Iou / len(dataset)))
    print("Mean Acc:" + str(Acc / len(dataset)))
    print("Mean Sen:" + str(Sen / len(dataset)))
    print("Mean Precision:" + str(Pre / len(dataset)))
main()