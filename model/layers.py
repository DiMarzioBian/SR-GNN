import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights


def make_layer(in_dim, out_dim, bin_size):
    if bin_size == 1:
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(bin_size, bin_size)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_dim, out_dim, bias=False),
            nn.Unflatten(1, (out_dim, 1, 1)),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(bin_size, bin_size)),
            nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_dim, momentum=.95),
            nn.ReLU(inplace=True))


class PyramidPoolingModule(nn.Module):
    """
    Pyramid pooling modules, takes bin_sizes to make parallel pooling layer
    """
    def __init__(self, opt, in_dim, out_dim):
        super(PyramidPoolingModule, self).__init__()
        self.bin_sizes = opt.bin_sizes
        self.w = opt.w
        self.h = opt.h
        self.num_label = opt.num_label

        self.modules_list = nn.ModuleList()
        self.modules_list = nn.ModuleList([make_layer(in_dim, out_dim, bin_size) for bin_size in self.bin_sizes])
        for module in self.modules_list:
            init_weights(module)

    def forward(self, in_feats):

        [h, w] = in_feats.shape[2:]
        out_feats = []
        for i, module in enumerate(self.modules_list):
            conv_feats = module(in_feats)
            out_feats.append(F.interpolate(input=conv_feats, size=[h, w], mode='bilinear', align_corners=False))

        out_feats.append(in_feats)
        # out_feats = [F.interpolate(stage(in_feats), size=[h, w], mode='bilinear') for stage in self.modules_list]

        return torch.cat(out_feats, 1)
