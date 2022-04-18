# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils import data
import torchvision
from torchvision import transforms
import pandas as pd

from gan_control.utils.logging_utils import get_logger

_log = get_logger(__name__)


class DataFrameDataSet(Dataset):
    def __init__(self, daraframe_path, attribute=None, train=True):
        _log.info('Loading dataframe from: %s' % daraframe_path)
        self.train = train
        self.attributes_df = pd.read_pickle(daraframe_path)
        if train:
            self.attributes_df = self.attributes_df.iloc[:int(self.__len__() * 0.9)]
        else:
            self.attributes_df = self.attributes_df.iloc[int(self.__len__() * 0.9):]
        self.attribute = attribute
        if self.attribute is not None:
            self.attributes_df = self.attributes_df[['latents_w', attribute]]
        _log.info('Dataset length: %d' % self.__len__())

    def __getitem__(self, index):
        attributes_series = self.attributes_df.iloc[index]
        if self.attribute in ['age']:
            attributes = torch.tensor(attributes_series[self.attribute]).unsqueeze(0)
        elif self.attribute in ['expression_q']:
            attributes = torch.nn.functional.one_hot(torch.tensor(attributes_series[self.attribute]), num_classes=8)
        else:
            attributes = torch.tensor(attributes_series[self.attribute])
        return attributes, torch.tensor(attributes_series['latents_w'])

    def __len__(self):
        return len(self.attributes_df.latents_w)


def get_dataframe_data_loader(daraframe_path, attribute, batch_size=32, shuffle=True, drop_last=True, workers=32, train=True):
    dataset = DataFrameDataSet(daraframe_path, attribute=attribute, train=train)
    _log.info('init dataframe data loader: batch size:%d, shuffle:%s, drop last:%s, num workers:%d' % (batch_size, str(shuffle), str(drop_last), workers))
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers
    )
    return loader
