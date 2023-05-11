#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:CASIA
# Time  :2019/4/10 18:36
import torch
from PIL import Image
import yaml
from easydict import EasyDict as edict
import numpy as np
import os
import sys
import ntpath
from scipy.misc import imresize
import warnings
warnings.filterwarnings("ignore")
if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

#Save images to the disk.
def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Log():

    def __init__(self, opt):
        self.opt = opt
        self.name = opt.name

    #print current losses on console; also save the losses to the disk
    def print_current_losses(self, epoch, iters, losses):

        message = '(Epoch: %d, iters: %d) loss: ' % (epoch, iters)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        # print the message
        print(message)

#Converts a Tensor array into a numpy image array.
def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy=np.transpose(image_numpy, (1, 2, 0))*0.5+0.5
        image_numpy = (image_numpy)* 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


#Save image to the disk
def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

#create directory
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#Load a config file and merge it into the default options
def cfg_from_file(filename):
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f.read()),Loader=yaml.SafeLoader)
    return yaml_cfg
#Merge config dictionary a into config dictionary b
def merge_a_into_b(a, b):
    if type(a) is not edict:
        return
    for k, v in a.items():
        if type(v) is edict:
            merge_a_into_b(a[k], b[k])
        else:
            b[k] = v