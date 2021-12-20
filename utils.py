import torch.nn as nn
import os
# import socket
# from dhooks import Webhook, Embed


def set_optimizer_lr(optimizer, lr):
    """Sets the learning rate to a specific one"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_optimizer_lr(optimizer):
    """Update the learning rate"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5


class Noter:
    def __init__(self, opt, webhook=False):
        self.log = opt.log
        self.loss_train = 1e8
        self.hr_train = 0
        self.mrr_train = 0
        self.ndcg_train = 0
        self.elapse = 1e8

        # set arguments into log file and data index
        with open(self.log, 'a') as f:
            for k, v in vars(opt).items():
                f.write('%s: %s\n' % (k, v))
            f.write('\nEpoch, Time, loss_tr, hr_tr, mrr_tr, ndcg_tr, loss_val, hr_val, mrr_val, ndcg_val\n')

        # log arguments in console
        print('\n[Info] Model settings:\n')
        for k, v in vars(opt).items():
            print('         %s: %s' % (k, v))

        # get own ip
        # if webhook:
        #     s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #     s.connect(("8.8.8.8", 80))
        #     self.ip = s.getsockname()[0]
        #     s.close()

    def log_train(self, loss, hr, mrr, ndcg, elapse):
        """ Print train stats in console. """
        self.loss_train = loss
        self.hr_train = hr
        self.mrr_train = mrr
        self.ndcg_train = ndcg
        self.elapse = elapse
        print('\n- (Training) Loss:{loss: 8.5f}, hr:{hr: 8.4f}, mrr:{mrr: 8.4f}, ndcg:{ndcg: 8.4f}, '
              'elapse:{elapse:3.4f} min'
              .format(loss=loss, hr=hr, mrr=mrr, ndcg=ndcg, elapse=elapse))

    def log_val(self, epoch, loss, hr, mrr, ndcg, set_log=True):
        """ Print validation stats in console. """
        if set_log:
            with open(self.log, 'a') as f:
                f.write('{epoch}, {time: 8.4f}, {loss_train: 8.5f}, {hr_train: 8.4f}, {mrr_train: 8.4f}, '
                        '{ndcg_train: 8.4f}, {loss_val: 8.5f}, {hr_val: 8.4f}, {mrr_val: 8.4f}, {ndcg_val: 8.4f}\n'
                        .format(epoch=epoch, time=self.elapse, loss_train=self.loss_train,
                                hr_train=self.hr_train, mrr_train=self.mrr_train, ndcg_train=self.ndcg_train,
                                loss_val=loss, hr_val=hr, mrr_val=mrr, ndcg_val=ndcg))

        print('\n- (Validating) Loss:{loss: 8.5f}, hr:{hr: 8.4f}, mrr:{mrr: 8.4f}, ndcg:{ndcg: 8.4f}'
              .format(loss=loss, hr=hr, mrr=mrr, ndcg=ndcg))

    def set_result(self, mode, loss, hr, mrr, ndcg, epoch_hr=None, epoch_mrr=None, epoch_ndcg=None):
        """ Set result of training or testing in log file and print in console. """
        if mode == 'train':
            with open(self.log, 'a') as f:
                f.write('\n[Info] Training stopped with best loss: {loss: 8.5f}, hr:{hr: 8.4f}, mrr:{mrr: 8.4f}, '
                        'ndcg:{ndcg: 8.4f}'
                        .format(loss=loss, hr=hr, mrr=mrr, ndcg=ndcg))

                if epoch_hr or epoch_mrr or epoch_ndcg:
                    f.write('\n[Info] Best hr at epoch: {epoch_hr: 3.0f}, mrr at epoch: {epoch_mrr: 3.0f}, '
                            'ndcg at epoch: {epoch_ndcg: 3.0f}'
                            .format(epoch_hr=epoch_hr, epoch_mrr=epoch_mrr, epoch_ndcg=epoch_ndcg))

            print('\n[Info] Training stopped with best loss: {loss: 8.5f}, hr:{hr: 8.4f}, mrr:{mrr: 8.4f}, '
                  'ndcg:{ndcg: 8.4f}'
                  .format(loss=loss, hr=hr, mrr=mrr, ndcg=ndcg))
                
            if epoch_hr or epoch_mrr or epoch_ndcg:
                print('\n[Info] Best hr at epoch: {epoch_hr: 3.0f}, mrr at epoch: {epoch_mrr: 3.0f}, '
                      'ndcg at epoch: {epoch_ndcg: 3.0f}'
                      .format(epoch_hr=epoch_hr, epoch_mrr=epoch_mrr, epoch_ndcg=epoch_ndcg))

        if mode == 'test':
            with open(self.log, 'a') as f:
                f.write('\n[Info] Test set result -> loss: {loss: 8.5f}, hr:{hr: 8.4f}, mrr:{mrr: 8.4f}, '
                        'ndcg:{ndcg: 8.4f}'
                        .format(loss=loss, hr=hr, mrr=mrr, ndcg=ndcg))

            print('\n[Info] Test set result -> Loss:{loss: 8.5f}, hr:{hr: 8.4f}, mrr:{mrr: 8.4f}, ndcg:{ndcg: 8.4f}'
                  .format(loss=loss, hr=hr, mrr=mrr, ndcg=ndcg))

    # def send_webhook(self, loss=0, hr=0, mrr=0, ndcg=0):
    #     name_project = os.getcwd().split('\\')[-1]
    #
    #     hook = Webhook('Insert discord webhook link')
    #     embed = Embed(
    #         description=name_project + ' finishes training.',
    #         color=0x00FF00,
    #         timestamp='now'
    #     )
    #     embed.set_author(name=self.ip)
    #     embed.add_field(name='loss', value=str(loss))
    #     embed.add_field(name='hr', value=str(hr))
    #     embed.add_field(name='mrr', value=str(mrr))
    #     embed.add_field(name='ndcg', value=str(ndcg))
    #     hook.send(embed=embed)

