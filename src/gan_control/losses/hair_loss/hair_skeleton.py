# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from collections import namedtuple
import yaml
import importlib
from torch.nn import functional as F
import warnings

from gan_control.losses.hair_loss.hair_model import PSPNet


class HairSkeleton(torch.nn.Module):
    def __init__(self, config):
        super(HairSkeleton, self).__init__()
        self.net = self.get_hair_model(config)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if x.shape[-1] != 256:
                x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
            mask = x.detach()
            with torch.no_grad():
                mask = self.normelize_to_model_input(mask)
                mask = self.net(mask)
                mask = self.make_mask_from_pred(mask)

            out = [torch.cat([x*mask, mask.float()], dim=1)]
        return out

    @staticmethod
    def make_mask_from_pred(pred):
        pred = torch.sigmoid(pred).detach()
        mask = pred >= 0.5
        return mask


    @staticmethod
    def get_hair_model(config):
        model = PSPNet(num_class=1, base_network='resnet101')
        model.load_state_dict(torch.load(config['model_path'])['weight'])
        model.eval()
        return model

    @staticmethod
    def normelize_to_model_input(batch):
        batch = batch.mul(0.5).add(0.5)
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        return (batch - mean) / std


