import os
from argparse import Namespace
from abc import ABC

from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


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
        seq, mask, gt = self.all_seq[index], self.mask[index], self.gt[index]
        items, num_item, A, alias_inputs = [], [], [], []
        for item in seq:
            num_item.append(len(np.unique(item)))
        max_num_item = np.max(num_item)

        for u_input in seq:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_num_item - len(node)) * [self.index_mask])
            u_A = np.zeros((max_num_item, max_num_item))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return torch.LongTensor(alias_inputs), torch.LongTensor(A), torch.LongTensor(items), torch.LongTensor(mask),\
               torch.LongTensor(gt)

    def get_data_mask(self, all_seq):
        """ Generate masked user sequences"""
        lens_all_seq = [len(seq) for seq in all_seq]
        len_max = max(lens_all_seq)
        all_seq_masked = [seq + [self.index_mask] * (len_max - len_seq) for seq, len_seq in zip(all_seq, lens_all_seq)]
        mask_all_seq = [[1] * len_seq + [self.index_mask] * (len_max - len_seq) for len_seq in lens_all_seq]
        return np.asarray(all_seq_masked), np.asarray(mask_all_seq), len_max


def get_sample_dataloader(opt: Namespace,
                          train_data: tuple,
                          test_data: tuple,
                          valid_data: tuple = (None, None)):
    """ Load data and prepare dataloader. """

    # Instancelize dataloader
    train_loader = DataLoader(SampleData(train_data), batch_size=opt.batch_size, num_workers=opt.num_workers,
                              shuffle=True)
    test_loader = DataLoader(SampleData(test_data), batch_size=opt.batch_size, num_workers=opt.num_workers,
                             shuffle=False)

    # Validation set
    if opt.validation:
        valid_loader = DataLoader(SampleData(valid_data), batch_size=opt.batch_size, num_workers=opt.num_workers,
                                shuffle=False)
    else:
        valid_loader = None

    return train_loader, test_loader, valid_loader
