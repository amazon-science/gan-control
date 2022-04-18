# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from collections import namedtuple
import yaml
import importlib
from torch.nn import functional as F
import pickle

from gan_control.losses.dogfacenet.models.pytorch_dogfacenet_model import DogFaceNet
from gan_control.utils.tensor_transforms import center_crop_tensor


class DogFaceNetSkeleton(torch.nn.Module):
    def __init__(self, config):
        super(DogFaceNetSkeleton, self).__init__()
        self.config = config
        self.net = self.get_dogfacenet_model(config)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x.mul(0.5).add(0.5)
        if x.shape[-1] != 224:
            if self.config['center_crop'] is not None:
                x = center_crop_tensor(x, self.config['center_crop'])
            x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=True)
        out = self.net(x)
        return [out]

    @staticmethod
    def get_dogfacenet_model(config):
        model = DogFaceNet()
        model.load_state_dict(torch.load(config['model_path']))
        model.eval()
        return model

    @staticmethod
    def normelize_to_model_input(batch):
        return batch


