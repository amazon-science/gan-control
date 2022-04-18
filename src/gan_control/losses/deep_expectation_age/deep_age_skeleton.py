# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from collections import namedtuple
import yaml
import importlib
import torchvision
from torch.nn import functional as F

from gan_control.losses.deep_expectation_age.deep_age_model import VGG
from gan_control.utils.tensor_transforms import center_crop_tensor


class DeepAgeSkeleton(torch.nn.Module):
    def __init__(self, config):
        super(DeepAgeSkeleton, self).__init__()
        self.config = config
        self.net = self.get_vgg(config)
        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def vgg_transform(x):
        x = x.mul(0.5).add(0.5)
        x[:,0,:,:] = x[:,0,:,:] - 0.48501961
        x[:,1,:,:] = x[:,1,:,:] - 0.45795686
        x[:,2,:,:] = x[:,2,:,:] - 0.40760392
        """Adapt image for vgg network, x: image of range(0,1) subtracting ImageNet mean"""
        r, g, b = torch.split(x, 1, 1)
        out = torch.cat((b, g, r), dim=1)
        out = F.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)
        out = out * 255.
        return out

    @staticmethod
    def get_predict_age(age_pb):
        predict_age_pb = F.softmax(age_pb)
        predict_age = torch.zeros(age_pb.size(0)).type_as(predict_age_pb)
        for i in range(age_pb.size(0)):
            for j in range(age_pb.size(1)):
                predict_age[i] += j * predict_age_pb[i][j]
        return predict_age

    def forward(self, x):
        if self.config['center_crop'] is not None:
            x = center_crop_tensor(x, self.config['center_crop'])
        x = self.vgg_transform(x)
        x = F.relu(self.net.conv1_1(x))
        x = F.relu(self.net.conv1_2(x))
        x = self.net.pool1(x)
        x = F.relu(self.net.conv2_1(x))
        x = F.relu(self.net.conv2_2(x))
        x = self.net.pool2(x)
        x = F.relu(self.net.conv3_1(x))
        x = F.relu(self.net.conv3_2(x))
        x = F.relu(self.net.conv3_3(x))
        x = self.net.pool3(x)
        x = F.relu(self.net.conv4_1(x))
        x = F.relu(self.net.conv4_2(x))
        x = F.relu(self.net.conv4_3(x))
        x = self.net.pool4(x)
        x = F.relu(self.net.conv5_1(x))
        x = F.relu(self.net.conv5_2(x))
        x = F.relu(self.net.conv5_3(x))
        x = self.net.pool5(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.net.fc6(x))
        out0 = F.relu(self.net.fc7(x))
        out1 = self.net.fc8_101(out0)


        return [out1]

    @staticmethod
    def get_vgg(config):
        model = VGG()
        vgg_state_dict = torch.load(config['model_path'])
        vgg_state_dict = {k.replace('-', '_'): v for k, v in vgg_state_dict.items()}
        model.load_state_dict(vgg_state_dict)
        model.eval()
        return model

