import os
import argparse
import numpy as np
import time
import pickle
import torch
import torch.nn as nn
from model.sr_gnn import SR_GNN
from datasets import getter_dataloader
from epoch import run_epoch
from utils import set_optimizer_lr, update_optimizer_lr, Noter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='1.0')
    parser.add_argument('--note', type=str, default='')

    # Model settings
    parser.add_argument('--hidden_size', type=int, default=50, help='hidden state size')
    parser.add_argument('--hybrid', action='store_false', default=True, help='if True, global + local; else global')
    parser.add_argument('--save_dict', type=bool, default=True)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--lr_step', type=int, default=20)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--es_patience', type=int, default=15)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)

    # Settings need to be tuned
    parser.add_argument('--dataset', default='sample')
    parser.add_argument('--step', type=int, default=2, help='Layer of GNN')
    parser.add_argument('--validation', default=False)
    parser.add_argument('--k_metric', type=int, default=20)

    # Add default args
    opt = parser.parse_args()
    opt.device = torch.device('cuda:0')
    opt.criterion = nn.CrossEntropyLoss()

    if opt.save_dict:
        opt.state_dict_path = '_result/model/v' + opt.version + time.strftime("-%b_%d_%H_%M", time.localtime()) + '.pkl'
    opt.log = '_result/v' + opt.version + time.strftime("-%b_%d_%H_%M", time.localtime()) + '.txt'

    if opt.dataset == 'diginetica':
        opt.num_item = 43097
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        opt.num_item = 37483
    elif opt.dataset == 'sample':
        opt.num_item = 309
    else:
        raise RuntimeError('Dataset ', str(opt.data), ' not found.')
    assert opt.k_metric <= opt.num_item

    """ Start modeling """
    noter = Noter(opt)

    # Import data
    data_getter = getter_dataloader(opt)

    # Load data
    print('\n[Info] Loading data...')
    trainloader, valloader, testloader = data_getter.get()

    # Load model
    model = SR_GNN(opt).to(opt.device)
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)

    # Loggers
    es_patience = 0
    loss_best = 1e9
    hr_best = 0
    mrr_best = 0
    ndcg_best = 0
    model_best = None

    for epoch in range(1, opt.epoch + 1):
        print('\n[ Epoch {epoch}]'.format(epoch=epoch))

        # Training
        start = time.time()
        loss_train, hr_train, ndcg_train, mrr_train = run_epoch(opt, model, trainloader, optimizer, mode='Train')
        scheduler.step()
        noter.log_train(loss=loss_train, hr=hr_train, mrr=mrr_train, ndcg=ndcg_train,
                        elapse=(time.time() - start)/60)

        # Validating
        with torch.no_grad():
            loss_val, hr_val, mrr_val, ndcg_val = run_epoch(opt, model, valloader, None, mode='Test')
        noter.log_val(epoch=epoch, loss=loss_val, hr=hr_val, mrr=mrr_val, ndcg=ndcg_val)

        # Early stopping
        if mrr_val > mrr_best or (mrr_val == mrr_best) & (ndcg_val <= ndcg_best):
            print("\n- New best performance logged.")
            es_patience = 0

            loss_best = loss_val
            hr_best = hr_val
            mrr_best = mrr_val
            ndcg_best = ndcg_val
            model_best = model.state_dict().copy()
        else:
            print("\n- Early stopping patience counter {} of {}".format(es_patience, opt.es_patience))
            es_patience += 1
            if es_patience == opt.es_patience:
                print("\n[Info] Stop training")
                break

    # Save stats and load best model
    noter.set_result(mode='train', loss=loss_best, hr=hr_best, mrr=mrr_best, ndcg=ndcg_best)
    model.load_state_dict(model_best)
    if opt.save_dict:
        with open(opt.state_dict_path, 'wb') as f:
            pickle.dump(model_best, f)

    """ Testing """
    with torch.no_grad():
        loss_test, hr_test, mrr_test, ndcg_test = run_epoch(opt, model, testloader, None, mode='Test')
    noter.set_result(mode='test', loss=loss_test, hr=hr_test, mrr=mrr_test, ndcg=ndcg_test)


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()


