import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_metrics(y_score, y_gt, num_label):
    """
    Compute mean IoU and per pixel accuracy.
    """
    y_pred = y_score.argmax(1)
    pa = y_pred.eq(y_gt).sum() / y_gt.numel()

    iou = torch.zeros(num_label).to(y_gt.device)
    for i in range(num_label):
        pred_i = (y_pred == i)
        gt_i = (y_gt == i)
        intersect = (pred_i & gt_i).sum()
        union = pred_i.sum() + gt_i.sum() - intersect
        if union < 0:
            x=1
        if union == 0:
            union = 1e9
        iou[i] = intersect / union

    return iou.mean(), pa




