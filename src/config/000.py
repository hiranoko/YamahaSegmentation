#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import os
from platform import architecture


version = f'000'
seed = 1111

n_fold = 5
num_workers = 2
target_size = (544, 1024) # None=Original size.
use_amp = True
use_mixup_cutmix = True
num_gpu = 1
batch_size = 2*num_gpu
normalize = False

with_soft_label = False

wo_mixup_epochs = 501
n_epochs = 15

pallet = {
    "non-traversable low vegetation" : [0, 160, 0],
    "sky" : [1, 88, 255],
    "high vegetation" : [40, 80, 0],
    "traversable grass" : [128, 255, 0],
    "rough trail" : [156, 76, 30],
    "smooth trail" : [178, 176, 153],
    "obstacle" : [255, 0, 0],
    "truck" : [255, 255, 255],
    "puddle" : [255, 0, 128]
}


project = '20220309_Segmentation'
input_dir = Path(f'/home/kota/work/{project}/yamaha_v0')
output_dir = Path(f'/home/kota/work/{project}/output')

# dataset
dataset_name = 'CustomDataset'

# model config
# model
model = dict(
    name = 'CustomModel',
    architecture = 'Unet',
    params = dict(
        encoder_name = 'resnet18',
        encoder_weights='imagenet',
        in_channels=3,
        classes=len(pallet),
    )
)

# optimizer
optim = dict(
    name = 'AdamW',
    lr = 0.001*num_gpu,
    weight_decay = 0.01
)

# loss
loss = dict(
    name = 'CustomLoss',
    params = dict(
    ),
)

# scheduler
scheduler = dict(
    name = 'CosineAnnealingLR',
    params = dict(
        T_max=n_epochs,
        eta_min=0,
        last_epoch=-1,
    )
)


# snapshot
snapshot = dict(
    save_best_only=True,
    mode='max',
    initial_metric=None,
    name=version,
    monitor='metric'
)

# logger
logger = dict(
    params=dict(
        logging_info=['loss', 'metric'],
        print=False
    ),
)

# augmentations.
horizontalflip = dict(
    name = 'HorizontalFlip',
    params = dict()
)

shiftscalerotate = dict(
    name = 'ShiftScaleRotate',
    params = dict(
        shift_limit = 0.1,
        scale_limit = 0.1,
        rotate_limit = 15,
    ),
)

gaussnoise = dict(
    name = 'GaussNoise',
    params = dict(
        var_limit = 5./255.
        ),
)

blur = dict(
    name = 'Blur',
    params = dict(
        blur_limit = 3
    ),
)

randommorph = dict(
    name = 'RandomMorph',
    params = dict(
        size = target_size,
        num_channels = 1,
    ),
)

randombrightnesscontrast = dict(
    name = 'RandomBrightnessContrast',
    params = dict(),
)

griddistortion = dict(
    name = 'GridDistortion',
    params = dict(),
)

elastictransform = dict(
    name = 'ElasticTransform',
    params = dict(
        sigma = 50,
        alpha = 1,
        alpha_affine = 10
    ),
)

totensor = dict(
    name = 'ToTensorV2',
    params = dict(),
)

oneof = dict(
    name='OneOf',
    params = dict(),
)

normalize = dict(
    name = 'Normalize',
    params = dict(),
)



# train.
train = dict(
    is_valid = False,
    data_path = input_dir / f'train_with_fold.csv',
    img_dir = input_dir / 'train',
    target_size = target_size,
    dataset_name = dataset_name,
    pallet = pallet,
    loader=dict(
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    transforms = [
        horizontalflip,
        #shiftscalerotate,
        #blur,
        #randombrightnesscontrast,
        normalize,
        totensor
        ],
)

# valid.
valid = dict(
    is_valid = True,
    data_path = input_dir / f'train_with_fold.csv',
    img_dir = input_dir / 'train',
    target_size = target_size,
    dataset_name = dataset_name,
    pallet = pallet,
    loader=dict(
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    transforms = [
        normalize,
        totensor
        ],
)

# test.
test = dict(
    is_valid = True,
    data_path = input_dir / 'test_with_fold.csv',
    img_dir = input_dir / 'photos',
    target_size = target_size,
    dataset_name = dataset_name,
    pallet = pallet,
    weight_name = f'{version}_best.pt',
    loader=dict(
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    transforms = [normalize,totensor],
)

