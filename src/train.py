#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')
import os
import sys
import cv2
import time
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler 
from torchvision.transforms import Normalize
from albumentations import Compose, OneOf, ShiftScaleRotate, HorizontalFlip
from albumentations.augmentations import transforms as aug
from kornia.metrics.mean_iou import mean_iou
#from albumentations.pytorch import ToTensor
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import random
from pathlib import Path
import gc

from utils import util
from utils.config import Config
import factory

class Runner:
    def __init__(self, cfg, model, criterion, optimizer, scheduler, device, logger):
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger

    def train(self, dataset_trn, dataset_val, loader_trn, loader_val):
        print(f'training start at {datetime.datetime.now()}')

        self.cfg.snapshot.output_dir = self.cfg.working_dir / 'weight'
        snap = util.Snapshot(**self.cfg.snapshot)

        for epoch in range(self.cfg.n_epochs):
            start_time = time.time()

            # train.
            result_trn = self.run_nn('trn', dataset_trn, loader_trn)
            
            # valid.
            with torch.no_grad():
                result_val = self.run_nn('val', dataset_val, loader_val)
            
            # scheduler step.
            if self.scheduler.__class__.__name__=='ReduceLROnPlateau':
                self.scheduler.step(result_val[self.cfg.scheduler.monitor])
            else:
                self.scheduler.step()

            wrap_time = time.time()-start_time

            # logging.
            logging_info = [epoch+1, wrap_time]
            logging_info.extend(sum([[result_trn[i], result_val[i]] for i in self.cfg.logger.params.logging_info], []))
            if self.logger:
                self.logger.write_log(logging_info)

            print(f"{epoch+1}/{self.cfg.n_epochs}: trn_loss={result_trn['loss']:.4f}, val_loss={result_val['loss']:.4f}, val_metric={result_val['metric']:.4f}, time={wrap_time:.2f}sec")

            # snapshot.
            snap.snapshot(result_val[self.cfg.snapshot.monitor], self.model, self.optimizer, epoch)

    def test(self, dataset_test, loader_test):
        print(f'test start at {datetime.datetime.now()}')
        with torch.no_grad():
            if self.cfg.with_soft_label:
                result = self.run_nn_with_soft_label('test', dataset_test, loader_test)
            else:
                result = self.run_nn('test', dataset_test, loader_test)
        print('done.')
        return result

    def run_nn(self, mode, dataset, loader):

        losses = []
        metrics = 0
        
        #sm = torch.nn.Sigmoid()

        if self.cfg.use_amp:
            scaler = GradScaler()

        if mode=='trn':
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
            val_metrics = np.zeros((len(dataset)))

        for idx, batch in enumerate(tqdm(loader)):
            img    = batch['image']
            target = batch['target']

            img   = img.to(self.device, dtype=torch.float)
            label = target.to(self.device, dtype=torch.float)

            # pred and calc losses.
            if self.cfg.use_amp:
                with autocast():
                    pred = self.model(img)
                    loss = self.criterion.calc_loss(pred, label)
            else:
                pred = self.model(img)
                loss = self.criterion.calc_loss(pred, label)
                
            losses.append(loss.item())

            if mode=='trn':
                if self.cfg.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                # make predictions.
                val_metrics[idx * self.cfg.batch_size:(idx + 1) * self.cfg.batch_size] += self.criterion.calc_metrics(pred, label).detach().item()

        if mode=='val':
            # calc. metrics.
            #val_pred = np.nan_to_num(val_pred)
            #val_pred[val_pred ==-np.inf] = 0
            #val_pred[val_pred == np.inf] = 0
            #metrics = self.criterion.calc_metrics(val_pred, val_target)
            
            metrics = np.average(val_metrics)
            #print(metric, val_metrics)
        elif mode=='test':
            return val_pred      

        result = dict(
            loss=np.sum(losses)/len(loader),
            metric=metrics,
        )

        return result

def train():
    args = util.get_args()
    print(args)
    cfg = Config.fromfile(args.config)
    cfg.fold = args.fold
    cfg.working_dir = cfg.output_dir / cfg.version / str(cfg.fold)
    print(f'version: {cfg.version}')
    print(f'fold: {cfg.fold}')

    # make output_dir if needed.
    util.make_output_dir_if_needed(cfg.working_dir)

    # set logger.
    cfg.logger.params.name = cfg.working_dir / f'history.csv'
    my_logger = util.CustomLogger(**cfg.logger.params)

    # set seed.
    util.seed_everything(cfg.seed)

    # get dataloader.
    print(f'dataset: {cfg.dataset_name}')
    folds = [fold for fold in range(cfg.n_fold) if cfg.fold != fold]
    dataset_train, loader_train = factory.get_dataset_loader(cfg.train, folds)
    dataset_valid, loader_valid = factory.get_dataset_loader(cfg.valid, [cfg.fold])

    # get model.
    print(f'model: {cfg.model.name}')
    model = factory.get_model(cfg.model)
    device = factory.get_device(args.gpu)
    model.cuda()
    model.to(device)
    
    if cfg.num_gpu>1:
        model = torch.nn.DataParallel(model)

    # get optimizer.
    print(f'optimizer: {cfg.optim.name}')
    plist = [{'params': model.parameters(), 'lr': cfg.optim.lr, 'weight_decay': cfg.optim.weight_decay}]
    optimizer = factory.get_optimizer(cfg.optim)(plist)

    # get loss.
    print(f'loss: {cfg.loss.name}')
    loss = factory.get_loss(cfg.loss)

    # get scheduler.
    print(f'scheduler: {cfg.scheduler.name}')
    scheduler = factory.get_scheduler(cfg.scheduler, optimizer)

    # run model.
    runner = Runner(cfg, model, loss, optimizer, scheduler, device, my_logger)
    runner.train(dataset_train, dataset_valid, loader_train, loader_valid)

if __name__ == '__main__':
    train()