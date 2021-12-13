

from model.metrics import *
from tqdm import tqdm


def run_epoch(opt,
              model,
              data,
              mode_train: bool = False,
              optimizer=None):

    num_data = data.dataset.length
    loss_epoch = 0
    hr_epoch = 0
    mrr_epoch = 0
    ndcg_epoch = 0

    if mode_train:
        model.train()
        mode = 'Train'
    else:
        model.eval()
        mode = 'Test'

    for batch in tqdm(data, desc='- ('+mode+'ing)   ', leave=False):
        # get data
        gt_batch = batch[-1].to(opt.device)
        scores_batch = model.predict(batch[:-1])

        # get loss
        loss_batch = opt.criterion(scores_batch, gt_batch - 1)
        if mode == 'Train':
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()

        # get metrics
        hr_batch, mrr_batch, ndcg_batch = get_metrics(scores_batch, gt_batch, opt.k_metric)

        ratio_batch = gt_batch.shape[0] / num_data

        loss_epoch += loss_batch * ratio_batch
        hr_epoch += hr_batch * ratio_batch
        mrr_epoch += mrr_batch * ratio_batch
        ndcg_epoch += ndcg_batch * ratio_batch

    return loss_epoch, hr_epoch, mrr_epoch, ndcg_epoch
