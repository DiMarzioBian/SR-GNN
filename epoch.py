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
        gt_batch = batch[-1].to(opt.device) - 1  # Item index GT starts from 1
        scores_batch = model.predict(batch[:-1])

        # get loss
        loss_batch = opt.criterion(scores_batch, gt_batch)
        if mode == 'Train':
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()

        # get metrics
        hits, mrr_sum, ndcg_sum = evaluate_result(scores_batch, gt_batch, opt.k_metric)

        loss_epoch += loss_batch * gt_batch.shape[0] / num_data
        hr_epoch += hits / num_data
        mrr_epoch += mrr_sum / num_data
        ndcg_epoch += ndcg_sum / num_data

    return loss_epoch, hr_epoch, mrr_epoch, ndcg_epoch
