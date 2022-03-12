#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2


class CustomDataset(Dataset):
    def __init__(self, cfg, folds, transforms):
        self.cfg = cfg
        self.transform = transforms

        self.df = pd.read_csv(self.cfg.data_path)
        self.df = self.df[self.df.fold.isin(folds)]
        
        # bellow is custom initializations.
        self.name = self.df.file_name.values
        
    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        img = cv2.imread(self.name[idx])
        ann = cv2.imread(self.name[idx].replace('rgb.jpg', 'labels.png'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
        
        masks = []
        for idx, (key, value) in enumerate(self.cfg.pallet.items()):
            mask = np.zeros((ann.shape[:2]), dtype=np.uint8)
            mask[np.where((ann==value).all(axis=2))] = 1
            masks.append(mask)
        
        if self.transform:
            augmented = self.transform(image=img, mask=np.stack(masks, axis=-1))
            img = augmented['image']
            mask = augmented['mask']

        mask = mask.permute(2,0,1)
            
        return {'image': img,
                'target': mask}
