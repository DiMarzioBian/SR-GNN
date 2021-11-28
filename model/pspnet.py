import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet import *
from model.layers import PyramidPoolingModule
from utils import init_weights


class Backbone(nn.Module):
    def __init__(self, backbone):
        super(Backbone,  self).__init__()
        if backbone == 'resnet18':
            self.model = resnet18(pretrained=True, replace_stride_with_dilation=[0, 2, 4])
        if backbone == 'resnet34':
            self.model = resnet34(pretrained=True, replace_stride_with_dilation=[0, 2, 4])
        if backbone == 'resnet50':
            self.model = resnet50(pretrained=True, replace_stride_with_dilation=[0, 2, 4])
        if backbone == 'resnet50':
            self.model = resnet50(pretrained=True, replace_stride_with_dilation=[0, 2, 4])
        if backbone == 'resnext50_32x4d':
            self.model = resnext50_32x4d(pretrained=True, replace_stride_with_dilation=[0, 2, 4])
        if backbone == 'wide_resnet50_2':
            self.model = wide_resnet50_2(pretrained=True, replace_stride_with_dilation=[0, 2, 4])


    def forward(self, x):
        x, x_auxiliary = self.model(x)
        return x, x_auxiliary


class Classifier(nn.Module):
    """
    Classifier for pyramid pooling features
    """
    def __init__(self, in_dim, num_label):
        super(Classifier,  self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_dim, num_label, kernel_size=(1, 1)),
        )
        init_weights(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class PSPNet(nn.Module):
    def __init__(self,
                 opt,
                 ):
        super(PSPNet, self).__init__()

        self.backbone = opt.backbone
        self.backbone_freeze = opt.backbone_freeze
        self.enable_aux = opt.enable_aux
        self.num_label = opt.num_label
        self.out_dim_pooling = opt.out_dim_pooling

        self.out_dim_resnet = opt.out_dim_resnet
        self.out_dim_resnet_auxiliary = opt.out_dim_resnet_auxiliary

        # Override Resnet official code, add dilation at BasicBlock 3 and 4 according to paper
        self.backbone = Backbone(self.backbone)
        if opt.backbone_freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.pyramid_pooling = PyramidPoolingModule(opt, in_dim=opt.out_dim_resnet, out_dim=self.out_dim_pooling)

        self.in_dim_classifier = opt.out_dim_resnet + len(opt.bin_sizes) * self.out_dim_pooling
        self.classifier = Classifier(in_dim=self.in_dim_classifier, num_label=self.num_label)

        if self.enable_aux:
            self.classifier_auxiliary = Classifier(in_dim=self.out_dim_resnet_auxiliary, num_label=self.num_label)

    def forward(self, img):
        if self.backbone_freeze:
            with torch.no_grad():
                x, x_auxiliary = self.backbone(img)
        else:
            x, x_auxiliary = self.backbone(img)
        x = self.pyramid_pooling(x)
        x = self.classifier(x)
        x = F.interpolate(x, img.shape[2:], mode='bilinear', align_corners=False)

        if self.enable_aux:
            x_auxiliary = self.classifier_auxiliary(x_auxiliary)
            x_auxiliary = F.interpolate(x_auxiliary, img.shape[2:], mode='bilinear', align_corners=False)
            return x, x_auxiliary
        else:
            return x, None

    def test(self, img):
        x, _ = self.backbone(img)
        x = self.pyramid_pooling(x)
        x = self.classifier(x)
        x = F.interpolate(x, img.shape[2:], mode='bilinear', align_corners=False)
        return x




