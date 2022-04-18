# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from collections import namedtuple
import yaml
import importlib
from torchvision import models
import torchvision
from torch.nn import functional as F

from gan_control.utils.tensor_transforms import center_crop_tensor


class StyleSkeleton(torch.nn.Module):
    def __init__(self, config):
        super(StyleSkeleton, self).__init__()
        self.net = models.vgg16(pretrained=True).features
        self.config = config
        self.resize_to = config['resize_to']
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), self.net[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), self.net[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), self.net[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), self.net[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # transformations = transforms.Compose([transforms.Scale(224),
        # transforms.CenterCrop(224), transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if x.shape[-1] != self.resize_to:
            x = F.interpolate(x, size=(self.resize_to, self.resize_to), mode='bilinear', align_corners=True)
        if self.config['center_crop'] is not None:
            x = center_crop_tensor(x, self.config['center_crop'])
        # normelize
        x = x.mul(0.5).add(0.5)
        x[:, 0, :, :] = (x[:, 0, :, :] - self.mean[0]) / self.std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - self.mean[1]) / self.std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - self.mean[2]) / self.std[2]

        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h

        features_style = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]
        features_style = [self.gram_matrix(y) for y in features_style]

        return features_style

    @staticmethod
    def gram_matrix(y):
        (b, ch, h, w) = y.shape
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


