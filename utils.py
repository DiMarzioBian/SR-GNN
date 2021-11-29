import torch.nn as nn
import os
import socket
from dhooks import Webhook, Embed


def set_optimizer_lr(optimizer, lr):
    """Sets the learning rate to a specific one"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_optimizer_lr(optimizer):
    """Update the learning rate"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5


def init_weights(model):
    """Initialize layer weights"""
    for layer in model:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.zero_()
        elif isinstance(layer, nn.BatchNorm2d):
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()


class Noter:
    def __init__(self, path_log, webhook=False):
        self.log = path_log
        self.loss_train = 1e8
        self.loss_aux_train = 1e8
        self.miou_train = 0
        self.pa = 0
        self.elapse = 1e8

        with open(self.log, 'a') as f:
            f.write('\nEpoch, Time, loss_tr, loss_aux_tr, miou_tr, pa_tr, loss_val, miou_val, pa_val\n')

        if webhook:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            self.ip = s.getsockname()[0]
            s.close()

    @staticmethod
    def set_args(opt):
        """ Set arguments into log file and log in console. """

        with open(opt.log, 'a') as f:
            for k, v in vars(opt).items():
                f.write('%s: %s\n' % (k, v))

        print('\n[Info] Model settings:\n')
        for k, v in vars(opt).items():
            print('         %s: %s' % (k, v))

    def log_train(self, loss, loss_aux, miou, pa, elapse):
        """ Print train stats in console. """

        self.loss_train = loss
        self.loss_aux_train = loss_aux
        self.miou_train = miou
        self.pa = pa
        self.elapse = elapse
        print('\n- (Training) Loss:{loss: 8.5f}, Loss_aux:{loss_aux: 8.5f}, mIoU:{miou: 8.4f}, pa:{pa: 8.4f}, '
              'elapse:{elapse:3.4f} min'
              .format(loss=loss, loss_aux=loss_aux, miou=miou, pa=pa, elapse=elapse))

    def log_val(self, epoch, loss, miou, pa, set_log=True):
        """ Print validation stats in console. """

        if set_log:
            with open(self.log, 'a') as f:
                f.write('{epoch}, {time: 8.4f}, {loss_train: 8.5f}, {loss_aux_train: 8.5f}, {miou_train: 8.4f}, '
                        '{pa_train: 8.4f}, {loss_val: 8.5f}, {miou_val: 8.4f}, {pa_val: 8.4f}\n'
                        .format(epoch=epoch, time=self.elapse, loss_train=self.loss_train,
                                loss_aux_train=self.loss_aux_train, miou_train=self.miou_train, pa_train=self.pa,
                                loss_val=loss, miou_val=miou, pa_val=pa))

        print('\n- (Validating) Loss:{loss: 8.5f}, mIoU:{miou: 8.4f}, pa:{pa: 8.4f}'
              .format(loss=loss, miou=miou, pa=pa))

    def set_train_result(self, loss, miou, pa):
        """ Set result of testloader and print in console. """

        with open(self.log, 'a') as f:
            f.write("\n[Info] Test set result -> loss: {loss: 8.5f}, miou: {miou: 8.4f} and pa: {pa: 8.4f}"
                    .format(loss=loss, miou=miou, pa=pa))

        print('\n- (Validating) Loss:{loss: 8.5f}, mIoU:{miou: 8.4f}, pa:{pa: 8.4f}'
              .format(loss=loss, miou=miou, pa=pa))

    def set_result(self, mode, loss, miou, pa):
        """ Set result of training or testing and print in console. """

        if mode == 'train':
            with open(self.log, 'a') as f:
                f.write("\n[Info] Training stopped with best loss: {loss: 8.5f}, best miou: {miou: 8.4f} "
                        "and best pa: {pa: 8.4f}"
                        .format(loss=loss, miou=miou, pa=pa))

            print("\n[Info] Training stopped with best loss: {loss: 8.5f}, best miou: {miou: 8.4f} "
                  "and best pa: {pa: 8.4f}\n"
                  .format(loss=loss, miou=miou, pa=pa))

        if mode == 'test':
            with open(self.log, 'a') as f:
                f.write("\n[Info] Test set result -> loss: {loss: 8.5f}, miou: {miou: 8.4f} and pa: {pa: 8.4f}"
                        .format(loss=loss, miou=miou, pa=pa))

            print('\n- (Validating) Loss:{loss: 8.5f}, mIoU:{miou: 8.4f}, pa:{pa: 8.4f}'
                  .format(loss=loss, miou=miou, pa=pa))

    def send_webhook(self, loss=0, miou=0, pa=0, name_project=None):
        if not name_project:
            name_project = os.getcwd().split('\\')[-1]

        hook = Webhook("x")
        embed = Embed(
            description=name_project + ' finishes training.',
            color=0x00FF00,
            timestamp='now'
        )
        embed.set_author(name=self.ip)
        embed.add_field(name='Loss', value=str(loss))
        embed.add_field(name='mIoU', value=str(miou))
        embed.add_field(name='Pixel acc', value=str(pa))
        hook.send(embed=embed)

