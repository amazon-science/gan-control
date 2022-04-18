# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from collections import namedtuple
import yaml
import importlib
from torch.nn import functional as F

from torchvision.models import resnet18
from gan_control.utils.tensor_transforms import center_crop_tensor


class ImageNetSkeleton(torch.nn.Module):
    def __init__(self, config):
        super(ImageNetSkeleton, self).__init__()
        self.config = config
        self.net = resnet18(pretrained=True)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.shape[-1] != 224:
            if self.config['center_crop'] is not None:
                x = center_crop_tensor(x, self.config['center_crop'])
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)

        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.net.avgpool(x)
        b_last = torch.flatten(x, 1)
        last = self.net.fc(b_last)

        return [last, b_last]

    @staticmethod
    def normelize_to_model_input(batch):
        return batch


if __name__ == '__main__':
    model = ImageNetSkeleton(None)

