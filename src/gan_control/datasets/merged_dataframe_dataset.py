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


class MergedDataFrameDataSet(Dataset):
    def __init__(self, daraframe_path, train=True):
        _log.info('Loading dataframe from: %s' % daraframe_path)
        self.train = train
        self.attributes_df = pd.read_pickle(daraframe_path)
        if train:
            self.attributes_df = self.attributes_df.iloc[:int(self.__len__() * 0.9)]
        else:
            self.attributes_df = self.attributes_df.iloc[int(self.__len__() * 0.9):]
        _log.info('Dataset length: %d' % self.__len__())

    def __getitem__(self, index):
        attributes_series = self.attributes_df.iloc[index]
        output_dict = {
            'arcface_emb': torch.tensor(attributes_series['arcface_emb']),
            'orientation': torch.tensor(attributes_series['orientation']),
            'gamma': torch.tensor(attributes_series['gamma3d']),
            'hair': torch.tensor(attributes_series['hair']),
            'age': torch.tensor(attributes_series['age']).unsqueeze(0),
            'expression': torch.tensor(attributes_series['expression3d']),
        }
        return output_dict, torch.tensor(attributes_series['latents_w'])

    def __len__(self):
        return len(self.attributes_df.latents_w)


def get_dataframe_data_loader(dataframe_path, batch_size=32, shuffle=True, drop_last=True, workers=32, train=True):
    dataset = MergedDataFrameDataSet(dataframe_path, train=train)
    _log.info('init dataframe data loader: batch size:%d, shuffle:%s, drop last:%s, num workers:%d' % (batch_size, str(shuffle), str(drop_last), workers))
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers
    )
    return loader

