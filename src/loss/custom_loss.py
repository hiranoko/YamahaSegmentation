import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.metrics.mean_iou import mean_iou

class CustomLoss:
    def __init__(self):
        self.loss = nn.BCEWithLogitsLoss()

    def calc_loss(self, preds, targets):
        return self.loss(preds, targets)

    def calc_metrics(self, preds, targets):
        batch_size = preds.shape[0]
        accuracy = torch.mean(
            mean_iou(
                preds.argmax(1).view(batch_size, -1),
                targets.argmax(1).view(batch_size, -1),
                num_classes=9
        ))
        return accuracy
