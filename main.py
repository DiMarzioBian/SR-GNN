import argparse
import numpy as np
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from model.pspnet import PSPNet
from datasets import getter_dataloader, get_data_detail
from epoch import train_epoch, test_epoch
from utils import set_optimizer_lr, update_optimizer_lr
import os
from dhooks import Webhook, Embed
import socket


def main():
    """
    Preparation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='1.0')
    parser.add_argument('--note', type=str, default='')

    # Model settings
    parser.add_argument('--shrink_image', type=list, default=[400, 600])
    parser.add_argument('--backbone_freeze', type=bool, default=True)
    parser.add_argument('--recalculate_mean_std', type=bool, default=False)
    parser.add_argument('--save_dict', type=bool, default=True)

    parser.add_argument('--lr_patience', type=int, default=20)
    parser.add_argument('--es_patience', type=int, default=15)
    parser.add_argument('--gamma_steplr', type=float, default=0.5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)

    # Settings need to be tuned
    parser.add_argument('--backbone', type=str, default='resnet18')  # Num of cross validation folds
    parser.add_argument('--data', default='assd')
    parser.add_argument('--bin_sizes', type=list, default=[1, 2, 3, 6])
    parser.add_argument('--id_optimizer', type=int, default=0)  # 0:AdamW, 1:Adam, 2:AMSGrad, 3:SGD + momentum, 4:SGD
    parser.add_argument('--enable_aux', type=bool, default=True)
    parser.add_argument('--alpha_loss', type=float, default=0.2)

    # Augmentation
    parser.add_argument('--enable_hvflip', type=float, default=0.5)  # enable horizontal and vertical flipping
    parser.add_argument('--enable_resize', type=float, default=0.5)  # apply white Gaussian noise

    opt = parser.parse_args()
    opt.log = '_result/v' + opt.version + time.strftime("-%b_%d_%H_%M", time.localtime()) + '.txt'
    opt.device = torch.device('cuda:0')

    if opt.save_dict:
        opt.state_dict_path = '_result/model/v' + opt.version + time.strftime("-%b_%d_%H_%M", time.localtime()) + '.pkl'

    # Model settings
    if opt.backbone == 'resnet18':
        opt.out_dim_resnet = 512
        opt.out_dim_resnet_auxiliary = 256
        opt.out_dim_pooling = 512
    elif opt.backbone == 'resnet34':
        opt.out_dim_resnet = 512
        opt.out_dim_resnet_auxiliary = 256
        opt.out_dim_pooling = 512
    elif opt.backbone == 'resnet50':
        opt.out_dim_resnet = 2048
        opt.out_dim_resnet_auxiliary = 1024
        opt.out_dim_pooling = 2048
    elif opt.backbone == 'resnext50_32x4d':
        opt.out_dim_resnet = 2048
        opt.out_dim_resnet_auxiliary = 1024
        opt.out_dim_pooling = 2048
    elif opt.backbone == 'wide_resnet50_2':
        opt.out_dim_resnet = 2048
        opt.out_dim_resnet_auxiliary = 1024
        opt.out_dim_pooling = 2048
    else:
        raise RuntimeError('\n[warning] Backbone not found.\n')

    opt.seg_criterion = nn.CrossEntropyLoss().to(opt.device)

    # Print hyperparameters and settings
    print('\n[Info] Model settings:\n')
    for k, v in vars(opt).items():
        print('         %s: %s' % (k, v))

    with open(opt.log, 'a') as f:
        # Save hyperparameters
        for k, v in vars(opt).items():
            f.write('%s: %s\n' % (k, v))

    """
    Start modeling
    """
    # Import data
    data_getter = getter_dataloader(opt)
    (opt.num_label, opt.h, opt.w) = get_data_detail(opt.data)

    with open(opt.log, 'a') as f:
        f.write('\nEpoch, Time, loss_tr, loss_aux_tr, miou_tr, pa_tr, loss_val, miou_val, pa_val\n')

    # Load model
    model = PSPNet(opt)
    model = model.to(opt.device)

    if opt.id_optimizer == 0:
        optimizer = optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-5)
    elif opt.id_optimizer == 1:
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-5)
    elif opt.id_optimizer == 2:
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-5,
                               amsgrad=True)
    elif opt.id_optimizer == 3:
        optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=1e-2, momentum=0.9,
                              weight_decay=1e-5, nesterov=True)
    elif opt.id_optimizer == 4:
        optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=1e-2, weight_decay=1e-5)
    else:
        raise RuntimeError('\n[warning] Optimizer out of index!\n')

    scheduler = optim.lr_scheduler.StepLR(optimizer, int(opt.lr_patience), gamma=opt.gamma_steplr)

    # Load data
    print('\n[Info] Loading data...')
    trainloader, valloader, testloader = data_getter.get()

    # Define logging variants
    loss_best = 1e9
    miou_best = 0
    pa_best = 0
    model_best = None

    for epoch in range(1, opt.epoch + 1):
        print('\n[ Epoch {epoch}]'.format(epoch=epoch))

        # """ Training """
        start = time.time()

        loss_train, loss_aux_train, miou_train, pa_train = train_epoch(model, trainloader, opt, optimizer)

        scheduler.step()

        end = time.time()

        print('\n- (Training) Loss:{loss: 8.5f}, Loss_aux:{loss_aux: 8.5f}, mIoU:{miou: 8.4f}, pa:{pa: 8.4f}, '
              'elapse:{elapse:3.4f} min'
              .format(loss=loss_train, loss_aux=loss_aux_train, miou=miou_train, pa=pa_train,
                      elapse=(time.time() - start) / 60))

        """ Validating """
        with torch.no_grad():
            loss_val, miou_val, pa_val = test_epoch(model, valloader, opt)

        print('\n- (Validating) Loss:{loss: 8.5f}, mIoU:{miou: 8.4f}, pa:{pa: 8.4f}'
              .format(loss=loss_val, miou=miou_val, pa=pa_val))

        """ Logging """
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {time: 8.4f}, {loss_train: 8.5f}, {loss_aux_train: 8.5f}, {miou_train: 8.4f}, '
                    '{pa_train: 8.4f}, {loss_val: 8.5f}, {miou_val: 8.4f}, {pa_val: 8.4f}\n'
                    .format(epoch=epoch, time=(end - start) / 60, loss_train=loss_train, loss_aux_train=loss_aux_train,
                            miou_train=miou_train, pa_train=pa_train, loss_val=loss_val, miou_val=miou_val,
                            pa_val=pa_val), )

        """ Early stopping """
        if miou_val > miou_best or (miou_val == miou_best) & (loss_val <= loss_best):
            loss_best = loss_val
            miou_best = miou_val
            pa_best = pa_val
            model_best = model.state_dict().copy()

            patience = 0
            print("\n- New best performance logged.")
        else:
            patience += 1
            print("\n- Early stopping patience counter {} of {}".format(patience, opt.es_patience))

            if patience == opt.es_patience:
                print("\n[Info] Stop training")
                break

    # Save state dict of best model
    with open(opt.state_dict_path, 'wb') as f:
        pickle.dump(model_best, f)

    print("\n[Info] Training stopped with best loss: {loss_best: 8.5f}, best miou: {miou_best: 8.4f} "
          "and best pa: {pa_best: 8.4f}\n"
          .format(loss_best=loss_best, miou_best=miou_best, pa_best=pa_best), )

    with open(opt.log, 'a') as f:
        f.write("\n[Info] Training stopped with best loss: {loss_best: 8.5f}, best miou: {miou_best: 8.4f} "
                "and best pa: {pa_best: 8.4f}"
                .format(loss_best=loss_best, miou_best=miou_best, pa_best=pa_best), )

    """ Testing """
    with torch.no_grad():
        loss_test, miou_test, pa_test = test_epoch(model, testloader, opt)

    print('\n- (Validating) Loss:{loss: 8.5f}, mIoU:{miou: 8.4f}, pa:{pa: 8.4f}'
          .format(loss=loss_test, miou=miou_test, pa=pa_test))

    with open(opt.log, 'a') as f:
        f.write("\n[Info] Test set result -> loss: {loss: 8.5f}, miou: {miou: 8.4f} and pa: {pa: 8.4f}"
                .format(loss=loss_test, miou=miou_test, pa=pa_test), )

    # Send discord webhook
    send_webhook(loss_test, miou_test, pa_test)


def send_webhook(loss=0, miou=0, pa=0, name_project=None):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    print(ip)

    if not name_project:
        name_project = os.getcwd().split('\\')[-1]

    hook = Webhook(
        "x")
    embed = Embed(
        description=name_project + ' finishes training.',
        color=0x00FF00,
        timestamp='now'
    )
    embed.set_author(name=ip)
    embed.add_field(name='Loss', value=str(loss))
    embed.add_field(name='mIoU', value=str(miou))
    embed.add_field(name='Pixel acc', value=str(pa))
    hook.send(embed=embed)


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()


