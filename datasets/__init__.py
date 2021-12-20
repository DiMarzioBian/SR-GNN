import pickle
import networkx as nx
from datasets.dataloader_sample import *
from sklearn.model_selection import train_test_split


class getter_dataloader(object):
    """
    Select dataset,

    Dataset
    """
    def __init__(self, opt):
        self.opt = opt
        dataset = self.opt.dataset

        if dataset == 'sample' or 1 == 1:
            self.get_dataset_dataloader = get_sample_dataloader
            dir_sample = '_data/' + dataset
            self.train_data = pickle.load(open(dir_sample + '/train.txt', 'rb'))
            self.test_data = pickle.load(open(dir_sample + '/test.txt', 'rb'))
            if opt.val_split_rate > 0:
                self.train_data, self.valid_data = split_validation(self.train_data, opt.val_split_rate)
            else:
                self.valid_data = None
        else:
            raise RuntimeError('Dataset ' + dataset + ' not found!')

    def get(self):
        """
        Return dataloader for train, validation and test
        """
        train_loader, val_loader, test_loader = self.get_dataset_dataloader(self.opt,
                                                                            self.train_data,
                                                                            self.test_data,
                                                                            self.valid_data)
        return train_loader, val_loader, test_loader


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)
