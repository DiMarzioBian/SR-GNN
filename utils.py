import torch.nn as nn


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
