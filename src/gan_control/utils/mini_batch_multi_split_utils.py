# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

import numpy as np

from gan_control.utils.logging_utils import get_logger

_log = get_logger(__name__)


class FcConfig:
    def __init__(self, in_order_group_names: List[str], groups):
        self.in_order_group_names: List[str] = in_order_group_names
        self.groups = groups


class MiniBatchUtils:
    def __init__(self, mini_batch, sub_groups_dict, total_batch=8, debug=False):
        self.mini_batch = mini_batch
        self.total_batch = total_batch
        self.sub_groups_dict = sub_groups_dict
        self.debug = debug
        self.num_of_sub_groups = len(self.sub_groups_dict)
        self.sub_group_names = self.get_ordered_group_names()
        self.place_in_mini_batch_dict = {}
        mini_batch_count = 0
        for name in self.sub_groups_dict.keys():
            self.place_in_mini_batch_dict[name] = self.sub_groups_dict[name]['place_in_mini_batch']
            if self.sub_groups_dict[name]['place_in_mini_batch'] is not None:
                mini_batch_count += self.sub_groups_dict[name]['place_in_mini_batch'][1] - self.sub_groups_dict[name]['place_in_mini_batch'][0]
        self.place_in_latent_dict = {}
        latent_count_size = 0
        for name in self.sub_groups_dict.keys():
            self.place_in_latent_dict[name] = self.sub_groups_dict[name]['place_in_latent']
            latent_count_size += self.sub_groups_dict[name]['place_in_latent'][1] - self.sub_groups_dict[name]['place_in_latent'][0]
        self.num_of_mini_batchs = total_batch // self.mini_batch
        if self.mini_batch != mini_batch_count:
            self.print()
            raise ValueError('self.mini_batch %d != mini_batch_count %d' % (self.mini_batch, mini_batch_count))
        if 512 != latent_count_size:
            self.print()
            raise ValueError('512 != latent_count_size %d' % latent_count_size)

    def get_ordered_group_names(self):
        start_latent_index = []
        latent_names = []
        for i, name in enumerate(self.sub_groups_dict.keys()):
            latent_names.append(name)
            start_latent_index.append(self.sub_groups_dict[name]['place_in_latent'][0])
        index_sorted_list = np.array(start_latent_index).argsort()
        ordered_group_names = [latent_names[n] for n in index_sorted_list]
        return ordered_group_names

    def get_sub_group(self, batch, sub_group_name='id'):
        return batch[self.place_in_mini_batch_dict[sub_group_name][0]:self.place_in_mini_batch_dict[sub_group_name][1]]

    def get_not_sub_group(self, batch, sub_group_name='id'):
        start_list = list(range(self.place_in_mini_batch_dict[sub_group_name][0]))
        end_list = list(range(self.place_in_mini_batch_dict[sub_group_name][1], self.mini_batch))
        return batch[start_list + end_list]

    def re_arrange_z(self, z_batch, batch_num):
        for group_name in self.sub_group_names:
            if self.place_in_mini_batch_dict[group_name] is not None:
                for i in range(self.place_in_mini_batch_dict[group_name][0], self.place_in_mini_batch_dict[group_name][1], 2):
                    z_batch[0][i + 1, self.place_in_latent_dict[group_name][0]:self.place_in_latent_dict[group_name][1]] = \
                        z_batch[0][i, self.place_in_latent_dict[group_name][0]:self.place_in_latent_dict[group_name][1]].detach()
        if len(z_batch) > 1:
            if 'other' in self.sub_group_names and self.place_in_mini_batch_dict['other'] is not None:
                for i in range(1, len(z_batch)):
                    z_batch[i][:self.place_in_mini_batch_dict['other'][0]] = z_batch[0][:self.place_in_mini_batch_dict['other'][0]]
                    z_batch[i][self.place_in_mini_batch_dict['other'][1]:] = z_batch[0][self.place_in_mini_batch_dict['other'][1]:]
            else:
                for i in range(1, len(z_batch)):
                    z_batch[i] = z_batch[0]
        return z_batch

    def extract_same_not_same_from_list(self, feature_list, same_group_name):
        same_list = []
        not_same_list = []
        for i in range(len(feature_list)):
            same_list.append(self.get_sub_group(feature_list[i], sub_group_name=same_group_name))
            not_same_list.append(self.get_not_sub_group(feature_list[i], sub_group_name=same_group_name))
        return same_list, not_same_list

    def print(self):
        text = 'MiniBatchUtils parameters:\n'
        text += 'mini batch size %d\n' % self.mini_batch
        text += 'total batch size %d\n' % self.total_batch
        text += 'sub group names %s\n' % str(self.sub_group_names)
        for i, group_name in enumerate(self.sub_group_names):
            text += '%d) %s: place in mini batch: %s place in latent: %s\n' % (i, group_name, str(self.place_in_mini_batch_dict[group_name]), str(self.place_in_latent_dict[group_name]))
        _log.info(text)

    def re_arrange_inject_noise(self, noises, group_name='id'):
        for i in range(self.place_in_mini_batch_dict[group_name][0], self.place_in_mini_batch_dict[group_name][1], 2):
            for j in range(len(noises)):
                noises[j][i+1, :, :, :] = noises[j][i, :, :, :].detach()
        return noises

    def get_fc_config(self):
        in_order_group_names = self.get_ordered_group_names()
        groups = self.get_groups()
        return FcConfig(in_order_group_names, groups)

    def get_groups(self):
        groups = {}
        for group_name in self.sub_group_names:
            groups[group_name] = {
                'latent_place': self.place_in_latent_dict[group_name],
                'latent_size': self.place_in_latent_dict[group_name][1] - self.place_in_latent_dict[group_name][0]
            }
        return groups


if __name__ == '__main__':
    from gan_control.trainers.utils import mixing_noise, make_mini_batch_from_noise

    class DefualtArgs(object):
        def __init__(self, batch_size, mini_batch_size):
            self.batch = batch_size
            self.mini_batch = mini_batch_size

    batch_size = 8
    mini_batch_size = 4
    same_id = 2
    same_pose = 2
    pose_id_split = 2
    latent_size = 4
    args = DefualtArgs(batch_size, mini_batch_size)
    noise = mixing_noise(batch_size, latent_size, 1., 'cuda')
    print(noise)
    mini_noise_inputs = make_mini_batch_from_noise(noise, args)





