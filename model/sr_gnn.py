import datetime
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def aggregate(self, hidden, A):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        gate_input = torch.sigmoid(i_i + h_i)
        gate_reset = torch.sigmoid(i_r + h_r)
        gate_output = torch.tanh(i_n + gate_reset * h_n)
        return gate_output + gate_input * (hidden - gate_output)

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.aggregate(A, hidden)
        return hidden


class SR_GNN(nn.Module):
    def __init__(self, opt):
        super(SR_GNN, self).__init__()
        # hyper parameters
        self.device = opt.device
        self.hidden_size = opt.hidden_size
        self.num_item = opt.num_item
        self.batch_size = opt.batch_size
        self.hybrid = opt.hybrid

        # network
        self.embedding = nn.Embedding(self.num_item, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc3 = nn.Linear(self.hidden_size, 1, bias=False)
        if self.hybrid:
            self.fc4 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()

        # init parameters
        std_var = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std_var, std_var)

    def forward(self, x, A):
        return self.gnn(self.embedding(x), A)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.fc1(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.fc2(hidden)  # batch_size x seq_length x latent_size
        alpha = self.fc3(torch.sigmoid(q1 + q2))
        pred_emb = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if self.hybrid:
            pred_emb = self.fc4(torch.cat([pred_emb, ht], 1))
        item_emb = self.embedding.weight[1:].transpose(1, 0)  # num_item x latent_size
        return torch.matmul(pred_emb, item_emb)

    def predict(self, data):
        seq_alias_batch, A_batch, items_batch, mask_batch = map(lambda x: x.to(self.device), data)
        num_sample = len(seq_alias_batch)

        hidden = self.forward(items_batch, A_batch)
        get = lambda i: hidden[i][seq_alias_batch[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(num_sample).long()])
        return self.compute_scores(seq_hidden, mask_batch)
