import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class FLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma


    def forward(self, logits, labels, neg_idx=None):
        
        probs = F.softmax(logits, dim=1)
        src = torch.ones(probs.shape[0]).cuda()
        src = torch.reshape(src, [src.shape[0], 1])
        labels = torch.reshape(labels, [labels.shape[0], 1])
        exp_labels = torch.zeros(probs.shape[0], 8, dtype=src.dtype).cuda().scatter_(1, labels, src)

        loss_pos = torch.pow((1.0 - probs), self.gamma) * exp_labels * torch.log(probs)
        probs = probs[:, 0]

        if neg_idx is not None:
            loss_neg = torch.pow((1.0 - probs), self.gamma) * neg_idx * torch.log(1.0 - probs + 1e-7)
            loss = torch.sum(loss_pos, dim=1) + loss_neg
        else:
            loss = torch.sum(loss_pos, dim=1)
        
        return -loss