# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from gan_control.models.gan_model import PixelNorm, EqualLinear
from gan_control.utils.logging_utils import get_logger

_log = get_logger(__name__)


class FcStack(nn.Module):
    def __init__(self, lr_mlp, n_mlp, in_dim, mid_dim, out_dim):
        super(FcStack, self).__init__()
        self.lr_mlp = lr_mlp
        self.n_mlp = n_mlp
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.fc_stack = self.create_input_middle_output_fc_stack(lr_mlp, n_mlp, in_dim, mid_dim, out_dim)

    @staticmethod
    def create_input_middle_output_fc_stack(lr_mlp, n_mlp, in_dim, mid_dim, out_dim):
        mid_dim = mid_dim if mid_dim is not None else mid_dim
        layers = []
        for i in range(n_mlp):
            s_dim0 = mid_dim
            s_dim1 = mid_dim
            if i == 0:
                s_dim0 = in_dim
            elif i == n_mlp - 1:
                s_dim1 = out_dim
            elif i < n_mlp - 1:
                pass
            else:
                raise ValueError('debug')
            layers.append(
                EqualLinear(
                    s_dim0, s_dim1, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        return nn.Sequential(*layers)

    def print(self):
        text = 'FcStack:\n'
        text += 'input dim: %d, middle dim:%d, output dim: %d\n' % (self.in_dim, self.mid_dim, self.out_dim)
        text += 'num of layers: %d\n' % (self.n_mlp)
        text += 'lr_mlp: %d' % (self.lr_mlp)
        _log.info(text)

    def forward(self, x):
        return self.fc_stack(x)