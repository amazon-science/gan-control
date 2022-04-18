# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from collections import namedtuple
import yaml
import importlib
import torchvision
from torch.nn import functional as F

from gan_control.losses.deep_head_pose.hopenet_model import Hopenet


class HopenetSkeleton(torch.nn.Module):
    def __init__(self, config):
        super(HopenetSkeleton, self).__init__()
        self.net = self.get_hopenet_model(config)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # transformations = transforms.Compose([transforms.Scale(224),
        # transforms.CenterCrop(224), transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        # normelize
        x = x.mul(0.5).add(0.5)
        x[:, 0, :, :] = (x[:, 0, :, :] - self.mean[0]) / self.std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - self.mean[1]) / self.std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - self.mean[2]) / self.std[2]


        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        layer1 = self.net.layer1(x)
        layer2 = self.net.layer2(layer1)
        layer3 = self.net.layer3(layer2)
        layer4 = self.net.layer4(layer3)

        x = self.net.avgpool(layer4)
        x = x.view(x.size(0), -1)
        pre_yaw = self.net.fc_yaw(x)
        pre_pitch = self.net.fc_pitch(x)
        pre_roll = self.net.fc_roll(x)

        output = torch.cat([pre_yaw.unsqueeze(1), pre_pitch.unsqueeze(1), pre_roll.unsqueeze(1)], dim=1)
        out = [layer1, layer2, layer3, layer4, output]

        return out

    @staticmethod
    def get_hopenet_model(config):
        model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        model.load_state_dict(torch.load(config['model_path']))
        model.eval()
        return model

