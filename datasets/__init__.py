import numpy as np
import random

from datasets.dataloader_assd import *
from sklearn.model_selection import train_test_split


class getter_dataloader(object):
    """
    Select dataset,

    Dataset
    """
    def __init__(self, opt):
        self.opt = opt
        dataset = self.opt.data

        if dataset == 'assd':
            dir_img = '_data/assd/original_images/'
            filename_list = []
            for _, _, file_list in os.walk(dir_img):
                for track_name in file_list:
                    filename_list.append(dir_img + track_name[:-4])
            self.filename_list = filename_list
            self.get_dataset_dataloader = get_assd_dataloader
        else:
            raise RuntimeError('Dataset ' + dataset + ' not found!')

    def get(self):
        """
        Return dataloader for train, validation and test
        """
        random.shuffle(self.filename_list)
        x = len(self.filename_list)

        train_index = self.filename_list[: int(x*0.8)]
        val_index = self.filename_list[int(x*0.8): int(x*0.9)]
        test_index = self.filename_list[int(x*0.9):]

        train_loader, val_loader, testloader = self.get_dataset_dataloader(self.opt, train_index, val_index, test_index)
        return train_loader, val_loader, testloader


def get_data_detail(dataset: str):
    if dataset == 'assd':
        return 24, 4000, 6000
    else:
        raise RuntimeError('Dataset ' + dataset + ' not found!')
