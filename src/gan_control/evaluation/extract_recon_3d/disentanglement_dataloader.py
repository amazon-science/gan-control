# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from itertools import chain
import os
import random

from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms


from igt_res_gan.utils.logging_utils import get_logger

_log = get_logger(__name__)


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DisentanglementDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(os.path.join(root))
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        im_name = os.path.split(fname)[1]
        uj = int(im_name.split('_')[0])
        ui = int(im_name.split('_')[1].split('.')[0])
        if self.transform is not None:
            img = self.transform(img)
        return img, uj, ui

    def __len__(self):
        return len(self.samples)


def get_disentanglement_data_loader(data_config, batch_size=4):
    compose_list = []
    compose_list.append(transforms.ToTensor())
    compose_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True))
    transform = transforms.Compose(compose_list)
    dataset = DisentanglementDataset(data_config['path'], transform=transform)
    shuffle = False
    _log.info('init Disentanglement data loader: batch size:%d, shuffle:%s, num workers:%d' % (batch_size, str(shuffle), data_config['workers']))
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=data_config['workers']
    )
    return loader
