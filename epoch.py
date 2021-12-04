import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from model.metrics import *


def train_epoch(model, data, optimizer, opt):
    """ Training """
    num_data = data.dataset.length
    loss_epoch = 0
    hr_epoch = 0
    mrr_epoch = 0
    ndcg_epoch = 0

    model.train()
    for batch in tqdm(data, desc='- (Training)   ', leave=False):
        # get data
        alias_seq, A_batch, items_batch, mask_batch, gt_batch = map(lambda x: x.to(opt.device), batch)
        items_batch = items_batch[0]

        # training model
        hidden = model(items_batch, A_batch)
        get = lambda i: hidden[i][alias_seq[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_seq)).long()])
        scores = model.compute_scores(seq_hidden, mask_batch)

        loss_batch = opt.seg_criterion(scores, gt_batch - 1)
        loss_batch.backward()
        optimizer.step()
        optimizer.zero_grad()

        # get metrics
        hr_batch, mrr_batch, ndcg_batch = get_metrics(scores, gt_batch - 1, opt.num_item)

        ratio_batch = alias_seq.shape[0] / num_data
        loss_epoch += loss_batch * ratio_batch
        hr_epoch += hr_batch * ratio_batch
        mrr_epoch += mrr_batch * ratio_batch
        ndcg_epoch += ndcg_batch * ratio_batch

    return loss_epoch, hr_epoch, mrr_epoch, ndcg_epoch


def test_epoch(model, data, opt):
    """ Testing """
    num_data = data.dataset.length
    loss_epoch = 0
    hr_epoch = 0
    mrr_epoch = 0
    ndcg_epoch = 0

    model.eval()
    for batch in tqdm(data, desc='- (Testing)   ', leave=False):
        # get data
        alias_seq, A_batch, items_batch, mask_batch, gt_batch = map(lambda x: x.to(opt.device), batch)
        items_batch = items_batch[0]

        # training model
        hidden = model(items_batch, A_batch)
        get = lambda i: hidden[i][alias_seq[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_seq)).long()])
        scores = model.compute_scores(seq_hidden, mask_batch)

        loss_batch = opt.seg_criterion(scores, gt_batch - 1)

        # get metrics
        hr_batch, mrr_batch, ndcg_batch = get_metrics(scores, gt_batch - 1, opt.num_item)

        ratio_batch = alias_seq.shape[0] / num_data
        loss_epoch += loss_batch * ratio_batch
        hr_epoch += hr_batch * ratio_batch
        mrr_epoch += mrr_batch * ratio_batch
        ndcg_epoch += ndcg_batch * ratio_batch

    return loss_epoch, hr_epoch, mrr_epoch, ndcg_epoch
