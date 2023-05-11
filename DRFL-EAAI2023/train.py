#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:CASIA
# Time  :2022/4/9 7:23
import  argparse
from util.util import *
import torch
import os,time
from dataset import dataset as datautils
from model import model_dcl as modelutils
from tqdm import tqdm
from util.Dice_test import *
parser = argparse.ArgumentParser()
parser.add_argument('-config', default='config.yml', help='config file')
parser.add_argument('-name', type=str, default='soft',help='name of the model.')
parser.add_argument('-gpu_ids', type=str, default='0', help='gpu ids: 0 / 0,1,2')
parser.add_argument('-output', type=str, default='./checkpoints', help='models are saved here')
parser.add_argument('-continue',action='store_true', help='continue_train ')
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

def main():
    f = open('log/'+a.name+".txt", 'w')
    if not os.path.exists(a.output):
        os.makedirs(a.output)
    for k, v in a.items():
        print(k, "=", v)
    train_data = datautils.datasetcreater(a,True)
    test_data = datautils.datasetcreater(a,False)
    # create a model given opt.model and other options
    model = modelutils.modelcreater(a,True)
    # regular setup: load and print networks; create schedulers
    model.setup(a)
    total_iters = 0  # the total number of training iterations
    mkdir(os.path.join(a.output, a.name))#mkdir for checkpoint
    best_dice=[0,0]
    for epoch in range(a.epoch_count,a.maintain_epoch + a.decay_epoch + 1):  # outer loop for different epochs
        train_bar = tqdm(train_data)
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        # inner loop within one epoch
        for data in train_bar:
            total_iters += a.batch_size
            epoch_iter += a.batch_size
            # unpack data from dataset and apply preprocessing
            model.set_input(data)
            # calculate loss functions, get gradients, update network weights
            model.optimize_parameters()
            losses = model.get_current_losses()
            train_bar.set_description(desc='[%d/%d] G_GAN loss: %.4f G_L1 loss: %.4f  D_real loss: %.4f  D_fake loss: %.4f  G_bin: %.4f  bin: %.4f' % (
                epoch,a.maintain_epoch + a.decay_epoch -a.epoch_count+1,losses['G_GAN'],losses['G_L1'],losses['D_real'],losses['D_fake'],losses['G_bin'],losses['bin']))
            #if total_iters % a.print_freq == 0:
                # print training losses and save logging information to the disk
             #   losses = model.get_current_losses()
              #  logutil.print_current_losses(epoch, epoch_iter, losses)

            if total_iters % a.save_latest_freq == 0:
                # cache  latest model
              #  print('Saving the latest model at (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if a.save_by_iter else 'latest'
                model.save_networks(save_suffix)


        if epoch % a.save_epoch_freq == 0:
            # cache  model
            model.save_networks('latest')
            model.save_networks(epoch)
      #  print("Dice:"+str(maxdice))
        # update learning rates at the end of every epoch.
        model.update_learning_rate()
        val_bar = tqdm(test_data)
        Dice = 0
        IOU=0
        for data in val_bar:
            # unpack data from data loader
            model.set_input(data)
            predict,groundtruth=model.test2()
            predict = tensor2im(predict)
            groundtruth = tensor2im(groundtruth)
            Dice+=getDice(groundtruth,predict,thresh=150)
        #    cv2.imwrite('res/' + str(getDice(groundtruth,predict))+'.png', predict)
            IOU+=getIoU(groundtruth,predict,thresh=150)
        val_bar.set_description(
            desc='[%d/%d] Dice : %.4f ; IOU : %.4f' % (
                epoch, a.maintain_epoch + a.decay_epoch - a.epoch_count + 1, Dice,IOU))
        print("Mean Dice:"+str(Dice/len(val_bar)))
        print("Mean IOU:" + str(IOU / len(val_bar)))
        f.write(str(epoch)+"Mean Dice:"+str(Dice/len(val_bar)))
        if best_dice[0]<Dice/len(val_bar):
            best_dice[0]=Dice/len(val_bar)
            best_dice[1]=epoch
            model.save_networks("best")
    print('Best Dice:'+str(best_dice[0])+" at Epoch:"+str(best_dice[1]))
    f.write('Best Dice:'+str(best_dice[0])+" at Epoch:"+str(best_dice[1]))
    f.close()



main()