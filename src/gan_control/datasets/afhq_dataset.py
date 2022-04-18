# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# this file borrows from https://github.com/clovaai/stargan-v2/blob/master/core/data_loader.py

from pathlib import Path
from itertools import chain
import os
import random

from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from gan_control.datasets.ffhq_dataset import data_sampler, sample_data
from gan_control.utils.logging_utils import get_logger

_log = get_logger(__name__)


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class AfhqDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(os.path.join(root, 'train', 'dog'))
        self.samples = self.samples + listdir(os.path.join(root, 'val', 'dog'))
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, (str(fname), str(fname))

    def __len__(self):
        return len(self.samples)


def get_afhq_data_loader(data_config, batch_size=4, size=512, training=True, prob=0.5):
    crop = transforms.RandomResizedCrop(size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(lambda x: crop(x) if random.random() < prob else x)
    compose_list = []
    if training: compose_list.append(rand_crop)
    compose_list.append(transforms.Resize([size, size]))
    if training: compose_list.append(transforms.RandomHorizontalFlip())
    compose_list.append(transforms.ToTensor())
    compose_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True))
    transform = transforms.Compose(compose_list)
    dataset = AfhqDataset(data_config['path'], transform=transform)
    shuffle = True
    drop_last = True
    _log.info('init AFHQ data loader: image size:%s, batch size:%d, shuffle:%s, drop last:%s, num workers:%d' % (size, batch_size, str(shuffle), str(drop_last), data_config['workers']))
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=data_sampler(dataset, shuffle=shuffle, distributed=False),
        drop_last=drop_last,
        num_workers=data_config['workers']
    )
    loader = sample_data(loader)
    return loader
