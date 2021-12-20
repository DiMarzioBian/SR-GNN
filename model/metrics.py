import numpy as np
import torch


def evaluate_result(scores, y_gt, k_metric):
    """ Get Hit rate, Mrr and NDCG """
    # metric
    hit, mrr, ndcg = 0, 0, 0
    idcg = 1 / np.log2(1 + 1)
    discount_dcg = np.log2(1+np.array(range(1, scores.shape[1]+1)))
    y_pred = scores.topk(k_metric)[1]

    for pred, gt in zip(y_pred, y_gt):
        res = torch.isin(pred, gt).int()
        if not res.sum() == 0:
            hit += 1
            res = res.cpu().detach().numpy()
            mrr += 1 / (1 + np.argmax(res))
            ndcg += 1/(discount_dcg[np.argmax(res)] * idcg)

    return hit, mrr, ndcg
