# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from collections import namedtuple
import yaml
import importlib
from torch.nn import functional as F

from gan_control.losses.arc_face.arc_face_model import Backbone, l2_norm
from gan_control.utils.tensor_transforms import center_crop_tensor


class ArcFaceSkeleton(torch.nn.Module):
    def __init__(self, config):
        super(ArcFaceSkeleton, self).__init__()
        self.config = config
        self.net = self.get_arc_face_model(config)
        self.layer1 = self.net.body[:3]
        self.layer2 = self.net.body[3:7]
        self.layer3 = self.net.body[7:21]
        self.layer4 = self.net.body[21:]
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.shape[-1] != 112:
            if self.config['center_crop'] is not None:
                x = center_crop_tensor(x, self.config['center_crop'])
            x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=True)
        x = self.net.input_layer(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        output = l2_norm(self.net.output_layer(layer4))
        out = [layer1, layer2, layer3, layer4, output]
        return out

    @staticmethod
    def get_arc_face_model(config):
        model = Backbone(config['num_layers'], config['drop_ratio'], mode=config['mode'])
        model.load_state_dict(torch.load(config['model_path']))
        model.eval()
        return model

    @staticmethod
    def normelize_to_model_input(batch):
        return batch


