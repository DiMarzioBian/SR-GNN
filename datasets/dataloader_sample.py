from argparse import Namespace
import numpy as np
import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader


class SampleData(Dataset):
    """ Sample dataset. """
    def __init__(self, data, shuffle=False, graph=None):
        self.index_mask = 0
        self.all_seq, self.mask, self.len_max = self.get_data_mask(data[0])
        self.gt = np.asarray(data[1])
        self.length = len(self.all_seq)
        self.shuffle = shuffle
        self.graph = graph

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.all_seq[index], self.mask[index], self.gt[index]

    def get_data_mask(self, all_seq):
        """ Generate masked user sequences"""
        lens_all_seq = [len(seq) for seq in all_seq]
        len_max = max(lens_all_seq)
        all_seq_masked = [seq + [self.index_mask] * (len_max - len_seq) for seq, len_seq in zip(all_seq, lens_all_seq)]
        mask_all_seq = [[1] * len_seq + [self.index_mask] * (len_max - len_seq) for len_seq in lens_all_seq]
        return np.asarray(all_seq_masked), np.asarray(mask_all_seq), len_max


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """
    seq_batch, mask_batch, gt_batch = list(zip(*insts))

    max_num_item_seq = np.max([len(np.unique(seq)) for seq in seq_batch])
    seq_alias_batch, items_batch, A_batch = [], [], []
    for seq in seq_batch:
        items_seq = np.unique(seq)
        items_batch.append(items_seq.tolist() + (max_num_item_seq - len(items_seq)) * [0])
        A_seq = np.zeros((max_num_item_seq, max_num_item_seq))  # Adjacency matrix for sequential
        for i in np.arange(len(seq) - 1):
            # For edges, seq[i] is in, and seq[i+1] is out
            if seq[i + 1] == 0:
                break
            in_index = np.where(items_seq == seq[i])[0][0]
            out_index = np.where(items_seq == seq[i + 1])[0][0]
            A_seq[in_index][out_index] = 1

        A_seq_in_sum = np.sum(A_seq, 0)
        A_seq_in_sum[np.where(A_seq_in_sum == 0)] = 1  # Add 1 for all nodes with 0 indegree
        A_seq_in = np.divide(A_seq, A_seq_in_sum)

        A_seq_out_sum = np.sum(A_seq, 1)
        A_seq_out_sum[np.where(A_seq_out_sum == 0)] = 1
        A_seq_out = np.divide(A_seq.transpose(), A_seq_out_sum)

        A_seq = np.concatenate([A_seq_in, A_seq_out]).transpose()
        A_batch.append(A_seq)
        seq_alias_batch.append([np.where(items_seq == i)[0][0] for i in seq])

    A_batch = torch.FloatTensor(np.array(A_batch))
    items_batch = torch.LongTensor(np.array(items_batch))
    seq_alias_batch = torch.LongTensor(np.array(seq_alias_batch))
    mask_batch = torch.LongTensor(np.array(mask_batch))
    gt_batch = torch.LongTensor(gt_batch)

    return A_batch, items_batch, seq_alias_batch, mask_batch, gt_batch


def get_sample_dataloader(opt: Namespace,
                          train_data: tuple,
                          test_data: tuple,
                          valid_data: tuple = (None, None)):
    """ Load data and prepare dataloader. """

    # Instancelize dataloader
    train_loader = DataLoader(SampleData(train_data), batch_size=opt.batch_size, num_workers=opt.num_workers,
                              collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(SampleData(test_data), batch_size=opt.batch_size, num_workers=opt.num_workers,
                             collate_fn=collate_fn, shuffle=False)

    # Validation set
    if opt.val_split_rate > 0:
        valid_loader = DataLoader(SampleData(valid_data), batch_size=opt.batch_size, num_workers=opt.num_workers,
                                  collate_fn=collate_fn, shuffle=False)
    else:
        valid_loader = test_loader

    return train_loader, valid_loader, test_loader
