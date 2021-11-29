import os
import argparse
import numpy as np
import time
import pickle
import torch
import torch.nn as nn
from model.sr_gnn import SR_GNN
from datasets import getter_dataloader, get_data_detail
from epoch import train_epoch, test_epoch
from utils import set_optimizer_lr, update_optimizer_lr, Noter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='1.0')
    parser.add_argument('--note', type=str, default='')

    # Model settings
    parser.add_argument('--shrink_image', type=list, default=[400, 600])
    parser.add_argument('--backbone_freeze', type=bool, default=True)
    parser.add_argument('--recalculate_mean_std', type=bool, default=False)
    parser.add_argument('--save_dict', type=bool, default=True)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_patience', type=int, default=20)
    parser.add_argument('--es_patience', type=int, default=15)
    parser.add_argument('--gamma_steplr', type=float, default=0.5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)

    # Settings need to be tuned
    parser.add_argument('--backbone', type=str, default='resnet18')  # Num of cross validation folds
    parser.add_argument('--data', default='assd')
    parser.add_argument('--alpha_loss', type=float, default=0.2)

    # Augmentation
    parser.add_argument('--enable_hvflip', type=float, default=0.5)  # enable horizontal and vertical flipping
    parser.add_argument('--enable_resize', type=float, default=0.5)  # apply white Gaussian noise

    # Add default args
    opt = parser.parse_args()
    opt.device = torch.device('cuda:0')
    if opt.save_dict:
        opt.state_dict_path = '_result/model/v' + opt.version + time.strftime("-%b_%d_%H_%M", time.localtime()) + '.pkl'
    opt.log = '_result/v' + opt.version + time.strftime("-%b_%d_%H_%M", time.localtime()) + '.txt'
    noter = Noter(opt.log)

    # Model settings
    if opt.dataset == 'diginetica':
        opt.n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        opt.n_node = 37484
    else:
        opt.n_node = 310

    """ Start modeling """
    noter.set_args(opt)

    # Import data
    data_getter = getter_dataloader(opt)
    (opt.num_label, opt.h, opt.w) = get_data_detail(opt.data)

    # Load model
    model = SR_GNN(opt).to(opt.device)

    # Load data
    print('\n[Info] Loading data...')
    trainloader, valloader, testloader = data_getter.get()

    # Define logging variants
    es_patience = 0
    loss_best = 1e9
    miou_best = 0
    pa_best = 0
    model_best = None

    for epoch in range(1, opt.epoch + 1):
        print('\n[ Epoch {epoch}]'.format(epoch=epoch))

        # Training
        start = time.time()
        loss_train, loss_aux_train, miou_train, pa_train = train_epoch(model, trainloader, opt)
        model.scheduler.step()
        noter.log_train(loss=loss_train, loss_aux=loss_aux_train, miou=miou_train, pa=pa_train,
                        elapse=(time.time() - start)/60)

        # Validating
        with torch.no_grad():
            loss_val, miou_val, pa_val = test_epoch(model, valloader, opt)
        noter.log_val(epoch=epoch, loss=loss_val, miou=miou_val, pa=pa_val)

        # Early stopping
        if miou_val > miou_best or (miou_val == miou_best) & (loss_val <= loss_best):
            loss_best = loss_val
            miou_best = miou_val
            pa_best = pa_val
            model_best = model.state_dict().copy()

            es_patience = 0
            print("\n- New best performance logged.")
        else:
            es_patience += 1
            print("\n- Early stopping patience counter {} of {}".format(es_patience, opt.es_patience))
            if es_patience == opt.es_patience:
                print("\n[Info] Stop training")
                break

    # Save stats and load best model
    noter.set_result(mode='train', loss=loss_best, miou=miou_best, pa=pa_best)
    model.load_state_dict(model_best)
    if opt.save_dict:
        with open(opt.state_dict_path, 'wb') as f:
            pickle.dump(model_best, f)

    """ Testing """
    with torch.no_grad():
        loss_test, miou_test, pa_test = test_epoch(model, testloader, opt)
    noter.set_result(mode='test', loss=loss_test, miou=miou_test, pa=pa_test)


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()


