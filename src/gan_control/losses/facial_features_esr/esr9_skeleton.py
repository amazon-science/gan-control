# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from collections import namedtuple
import yaml
import importlib
import torchvision
from torch.nn import functional as F

from gan_control.losses.facial_features_esr.esr9_model import ESR
from gan_control.utils.tensor_transforms import center_crop_tensor


class ESR9Skeleton(torch.nn.Module):
    def __init__(self, config):
        super(ESR9Skeleton, self).__init__()
        self.config = config
        self.net = self.get_esr9_model(config)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.config['center_crop'] is not None:
            x = center_crop_tensor(x, self.config['center_crop'])
        if x.shape[-1] != ESR.INPUT_IMAGE_SIZE[0]:
            x = F.interpolate(x, size=ESR.INPUT_IMAGE_SIZE, mode='bilinear', align_corners=True)
        # normelize
        x = x.mul(0.5).add(0.5)

        emotions = []
        affect_values = []

        # Get shared representations
        x_shared_representations = self.net.base(x)
        for branch in self.net.convolutional_branches:
            output_emotion, output_affect = branch(x_shared_representations)
            emotions.append(output_emotion.unsqueeze(1))
            affect_values.append(output_affect)

        out = [x_shared_representations, torch.cat(emotions, dim=1)]
        return out

    @staticmethod
    def get_esr9_model(config):
        model = ESR(config['model_path'])
        model.eval()
        return model


if __name__ == '__main__':
    model = ESR()
    print(model)

