#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:CASIA
# Time  :2019/8/2 15:18
import numpy as np
import os
import cv2
def Dice_test(filepath):
    n_test = int(len(os.listdir(filepath)) / 3)
    ious = np.zeros(n_test)
    dices = np.zeros(n_test)
    i = 0
    for filename in os.listdir(filepath):
        if 'groundtruth' in filename:
            filenamepre = filename.split('_')[0] + "_predict.png"
            gt = cv2.imread(filepath + "/" + filename)
            pr = cv2.imread(filepath + "/" + filenamepre)
            gt = cv2.split(gt)[0]
            pr = cv2.split(pr)[0]
            rw,gt = cv2.threshold(gt, 0, 255, cv2.THRESH_BINARY)
            rw,pr = cv2.threshold(pr, 0, 255, cv2.THRESH_BINARY)
            #gt = gt > 0
            #pr = pr > 0
            dices[i] = Dice(gt, pr)
            ious[i] = IoU(gt, pr)
            i += 1
          #  print(str(i) + "  " + filename + " Dice:" + str(Dice(gt, pr)) + "  IOU:" + str(IoU(gt, pr)))
        else:
            continue

    print('Mean IoU:', ious.mean())
    print('Mean Dice:', dices.mean())
    return  dices.mean()
def IoU(y_true, y_pred):
    #Returns Intersection over Union score for ground truth and predicted masks.
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def Dice(y_true, y_pred):
    #Returns Dice Similarity Coefficient for ground truth and predicted masks.
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)
def getDice(gt,pr,thresh=128):
    rw, gt = cv2.threshold(gt, 128, 255, cv2.THRESH_BINARY)
    rw, pr = cv2.threshold(pr, thresh, 255, cv2.THRESH_BINARY)
    gt = gt > 0
    pr = pr > 0
    return Dice(gt, pr)

def getIoU(gt,pr,thresh=128):
    rw, gt = cv2.threshold(gt, 128, 255, cv2.THRESH_BINARY)
    rw, pr = cv2.threshold(pr, thresh, 255, cv2.THRESH_BINARY)
    gt = gt > 0
    pr = pr > 0
    return IoU(gt, pr)

def getDiceselect(gt,pr,thresh):
    rw, gt = cv2.threshold(gt, 128, 255, cv2.THRESH_BINARY)
    rw, pr = cv2.threshold(pr, thresh, 255, cv2.THRESH_BINARY)
    gt = gt > 0
    pr = pr > 0
    return Dice(gt, pr)

def getIoUselect(gt,pr,thresh):
    rw, gt = cv2.threshold(gt, 128, 255, cv2.THRESH_BINARY)
    rw, pr = cv2.threshold(pr, thresh, 255, cv2.THRESH_BINARY)
    gt = gt > 0
    pr = pr > 0
    return IoU(gt, pr)