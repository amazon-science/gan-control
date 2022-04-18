# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from collections import namedtuple
import yaml
import importlib
from torch.nn import functional as F

from gan_control.losses.face3dmm_recon.models.pytorch_3d_recon_model import Recon3D
from gan_control.utils.tensor_transforms import center_crop_tensor


class Face3dmmSkeleton(torch.nn.Module):
    def __init__(self, config):
        super(Face3dmmSkeleton, self).__init__()
        self.config = config
        self.net = self.get_face_3dmm_model(config)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        r = x[:, :1, :, :]
        g = x[:, 1:2, :, :]
        b = x[:, 2:3, :, :]
        x = torch.cat([b, g, r], dim=1).mul(0.5).add(0.5).mul(255)
        if x.shape[-1] != 224:
            if self.config['center_crop'] is not None:
                x = center_crop_tensor(x, self.config['center_crop'])
            x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=True)
        out = self.net(x)
        return [out]

    @staticmethod
    def extract_futures_from_vec(out_vec):
        out_vec = out_vec[-1]
        return [out_vec[:,:80]], [out_vec[:,80:144]], [out_vec[:,144:224]], [out_vec[:,224:227]], [out_vec[:,227:254]], [out_vec[:,254:256]], [out_vec[:,256:257]]

    @staticmethod
    def get_face_3dmm_model(config):
        model = Recon3D()
        model.load_state_dict(torch.load(config['model_path']))
        model.eval()
        return model

    @staticmethod
    def normelize_to_model_input(batch):
        return batch


