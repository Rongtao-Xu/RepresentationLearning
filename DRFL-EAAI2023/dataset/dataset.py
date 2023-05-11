#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:CASIA
# Time  :2021/7/31 15:45

import os,glob
import torch
import os.path
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

#Interface
def datasetcreater(opt,isTrain):
    if isTrain==True:
        data_loader = DatasetDataLoader(opt,isTrain)
        dataset = data_loader.load_data()
    else:
        data_loader = DatasetDataLoaderTest(opt, isTrain)
        dataset = data_loader.load_data()
    return dataset

#Generate DatasetLoader
class DatasetDataLoaderTest():
    def __init__(self, opt,isTrain):
        self.opt = opt
        self.batch_size=1
        self.dataset = GetDataset(opt,isTrain)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        #Return the number of data in the dataset
        return int(len(self.dataset)/int(self.batch_size))

    def __iter__(self):
        #Return a batch of data
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

#Generate DatasetLoader
class DatasetDataLoader():
    def __init__(self, opt,isTrain):
        self.opt = opt
        self.batch_size=opt.batch_size
        self.dataset = GetDataset(opt,isTrain)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        #Return the number of data in the dataset
        return int(len(self.dataset)/int(self.batch_size))

    def __iter__(self):
        #Return a batch of data
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

            # prepare dataset


#Generate Dataset
class GetDataset(data.Dataset):

    def __init__(self, opt,isTrain):
        self.opt = opt
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc
        self.input = opt.input  # get the image directory
        self.proir=opt.proir
        if isTrain == True:
            phase='train'
            self.target = opt.groundtruth
              # get the image directory
        else:
            phase = 'test'
            self.target = opt.groundtruth  # get the image directory
        self.dir_A = os.path.join(opt.data_path,phase,self.input)  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.data_path,phase,self.target)  # create a path '/path/to/data/trainB'
        self.A_paths = load_impath(self.dir_A, opt.max_dataset_size)  # get image paths
        self.B_paths = load_impath(self.dir_B, opt.max_dataset_size)  # get image paths
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.dir_C = os.path.join(opt.data_path, phase, self.proir)
        self.C_paths = load_impath(self.dir_C, opt.max_dataset_size)
        if self.input_nc>3:
            self.dir_A2=os.path.join(opt.data_path,phase,opt.withsoft)
            self.A2_paths = load_impath(self.dir_A2, opt.max_dataset_size)
    # Return a data point and its metadata information.
    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        transform3C = my_transform(self.opt, grayscale=False)
        transform2C = my_transform(self.opt, grayscale=True)
        transform1C = my_transformsr(self.opt, grayscale=True)
        A = transform3C(A_img)
        B = transform2C(B_img)
        C_path = self.C_paths[index % self.B_size]
        C_img = Image.open(C_path).convert('RGB')
        C = transform1C(C_img)

        if self.input_nc>3:
            A2_path = self.A2_paths[index % self.B_size]
            A2_img = Image.open(A2_path).convert('RGB')
            A2=transform1C(A2_img)
            A=torch.cat((A,A2),dim=0)
        return {'A': A, 'B': B, 'C': C,'A_paths': A_path, 'B_paths': B_path}


    # number of images
    def __len__(self):
        return max(self.A_size, self.B_size)


#read images path | png can replace by other images type in here
def load_impath(dir, max_dataset_size):
    if dir is None or not os.path.exists(dir):
        raise Exception("input_dir does not exist")

  #  print(dir)
    input_paths = glob.glob(os.path.join(dir, "*.*"))
    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name
    #If sorting is a pure number, sort by the value of the number
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)
    return input_paths[:min(float(max_dataset_size), len(input_paths))]

#get image transform steps
def get_transform(opt, grayscale=False, method=Image.BICUBIC):
    transform_list = []
    new_h = new_w = opt.load_size
    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))
    flip = random.random() > 0.5
    params={'crop_pos': (x, y), 'flip': flip}
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    osize = [opt.load_size, opt.load_size]
    transform_list.append(transforms.Resize(osize, method))
    transform_list.append(transforms.Lambda(lambda img: new_crop(img, params['crop_pos'], opt.crop_size)))
    if not opt.no_flip:
        if params['flip']:
            transform_list.append(transforms.Lambda(lambda img: new_flip(img, params['flip'])))
    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)
#get test
def my_transform(opt, grayscale=False, method=Image.BICUBIC):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    osize = [opt.crop_size, opt.crop_size]
    transform_list.append(transforms.Resize(osize, method))
   # transform_list += [transforms.ToTensor()]
    transform_list += [transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))]
    return transforms.Compose(transform_list)

def my_transformsr(opt, grayscale=False, method=Image.BICUBIC):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    osize = [512, 512]
    transform_list.append(transforms.Resize(osize, method))
   # transform_list += [transforms.ToTensor()]
    transform_list += [transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))]
    return transforms.Compose(transform_list)


#define transform step crop
def new_crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

#define transform step flip
def new_flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img









