# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO

from PIL import Image
from torch.utils.data import Dataset
from torch.utils import data
import torchvision
from torchvision import transforms

from gan_control.utils.logging_utils import get_logger

_log = get_logger(__name__)


class FfhqData(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(FfhqData, self).__init__(root, transform=transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, (target, path)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def get_ffhq_data_loader(data_config, batch_size=4, size=1024, training=True):
    compose_list = []
    if size != 1024:
        compose_list.append(transforms.Resize(size))
    if training:
        compose_list.append(transforms.RandomHorizontalFlip())
    compose_list.append(transforms.ToTensor())
    compose_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True))
    transform = transforms.Compose(compose_list)

    #dataset = torchvision.datasets.ImageFolder(data_config['path'], transform=transform)
    dataset = FfhqData(data_config['path'], transform=transform)
    shuffle = True
    drop_last = True
    _log.info('init FFHQ data loader: image size:%s, batch size:%d, shuffle:%s, drop last:%s, num workers:%d' % (size, batch_size, str(shuffle), str(drop_last), data_config['workers']))
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=data_sampler(dataset, shuffle=shuffle, distributed=False),
        drop_last=drop_last,
        num_workers=data_config['workers']
    )
    loader = sample_data(loader)
    return loader
