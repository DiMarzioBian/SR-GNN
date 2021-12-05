import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_metrics(scores_batch, gt_batch, k_metric):
    """ Get Hit rate, Mrr and NDCG """
    hr, mrr, ndcg = [], [], []
    idcg = 1 / np.log2(1 + 1)
    discount_dcg = np.log2(1+np.array(range(1, scores_batch.shape[1]+1)))
    pred_batch = scores_batch.topk(k_metric)[1]

    for i in range(pred_batch.shape[0]):
        for pred, target in zip(pred_batch, gt_batch):
            res = torch.isin(pred, target).int().cpu().detach().numpy()

            # Not hit
            if np.sum(res) == 0:
                hr.append(0)
                mrr.append(0)
                ndcg.append(0)
            # hit
            else:
                hr.append(1)
                mrr.append(1 / (1 + np.argmax(res)))
                ndcg.append(1/(discount_dcg[np.argmax(res)] * idcg))

    return np.mean(hr), np.mean(mrr), np.mean(ndcg)
