#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch import Unet

class CustomModel(nn.Module):
    def __init__(self, architecture, params):
        super(CustomModel, self).__init__()

        self.model = eval(architecture)(
            encoder_name = params.encoder_name,
            encoder_weights=params.encoder_weights,
            in_channels=params.in_channels,
            classes=params.classes
        )
        
    def forward(self, x):
        x = self.model(x)
        return x
