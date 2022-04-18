# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np

from gan_control.utils.mini_batch_multi_split_utils import MiniBatchUtils, FcConfig
from gan_control.utils.logging_utils import get_logger

_log = get_logger(__name__)


class RandomMiniBatchUtils(MiniBatchUtils):
    def __init__(self, mini_batch, sub_groups_dict, total_batch=8, debug=False):
        super(RandomMiniBatchUtils, self).__init__(mini_batch, sub_groups_dict, total_batch=total_batch, debug=debug)
        self.checks()
        self.count_in_mini_bach_dict = {}
        for name in self.sub_groups_dict.keys():
            self.count_in_mini_bach_dict[name] = self.sub_groups_dict[name]['count_in_mini_bach']
        self.randomize_places_in_batch()

    def checks(self):
        if self.mini_batch != self.total_batch:
            raise ValueError('RandomMiniBatchUtils has mini_batch %d vs. total_batch %d' % (self.mini_batch, self.total_batch))

    def get_sub_group(self, batch, sub_group_name='id'):
        new_batch = []
        place_in_mini_batch = self.place_in_mini_batch_dict[sub_group_name]
        for place in place_in_mini_batch:
            new_batch.append(batch[place[0]:place[1]])
        return torch.cat(new_batch, dim=0)

    def get_not_sub_group(self, batch, sub_group_name='id'):
        new_batch = []
        place_not_in_mini_batch = self.place_in_mini_batch_dict[sub_group_name]
        last_place = 0
        for i, place in enumerate(place_not_in_mini_batch):
            if last_place != place[0]:
                new_batch.append(batch[last_place:place[0]])
            last_place = place[1]
        if last_place != (len(place_not_in_mini_batch) - 1):
            new_batch.append(batch[last_place:])
        return torch.cat(new_batch, dim=0)

    def randomize_places_in_batch(self):
        for group_name in self.sub_group_names:
            if self.count_in_mini_bach_dict[group_name] is not None:
                group_size = np.random.choice(
                    np.arange(self.count_in_mini_bach_dict[group_name][0], self.count_in_mini_bach_dict[group_name][1] + 2, 2),
                    1
                )
                places = None
                if group_size > 0:
                    start_places = list(np.random.choice(
                        np.arange(0, self.mini_batch, 2),
                        group_size // 2,
                        replace=False
                    ))
                    places = []
                    start_places.sort()
                    for i in range(len(start_places)):
                        if i != 0 and places[-1][1] == start_places[i]:
                            places[-1][1] = places[-1][1] + 2
                        else:
                            places.append([start_places[i], start_places[i] + 2])
                self.place_in_mini_batch_dict[group_name] = places
        if self.debug:
            self.print_places_in_min_batch()

    def re_arrange_z(self, z_batch, batch_num):
        for group_name in self.sub_group_names:
            if self.place_in_mini_batch_dict[group_name] is not None:
                for place_in_mini_batch in self.place_in_mini_batch_dict[group_name]:
                    for i in range(place_in_mini_batch[0], place_in_mini_batch[1], 2):
                        z_batch[0][i + 1, self.place_in_latent_dict[group_name][0]:self.place_in_latent_dict[group_name][1]] = \
                            z_batch[0][i, self.place_in_latent_dict[group_name][0]:self.place_in_latent_dict[group_name][1]].detach()
        if len(z_batch) > 1:
            raise ValueError('len(z_batch) = %d > 1, RandomMiniBatchUtils is not suppurting mixing right now, need to implement' % len(z_batch))
            # if 'other' in self.sub_group_names and self.place_in_mini_batch_dict['other'] is not None:
            #     for i in range(1, len(z_batch)):
            #         z_batch[i][:self.place_in_mini_batch_dict['other'][0]] = z_batch[0][:self.place_in_mini_batch_dict['other'][0]]
            #         z_batch[i][self.place_in_mini_batch_dict['other'][1]:] = z_batch[0][self.place_in_mini_batch_dict['other'][1]:]
            # else:
            #     for i in range(1, len(z_batch)):
            #         z_batch[i] = z_batch[0]
        return z_batch

    def print_places_in_min_batch(self):
        text = 'RandomMiniBatchUtils random places in mini batch:\n'
        for i, group_name in enumerate(self.sub_group_names):
            text += '%d) %s: count in mini batch: %s place in latent: %s\n' % (i, group_name, str(self.place_in_mini_batch_dict[group_name]), str(self.place_in_latent_dict[group_name]))
        _log.info(text)


    def print(self):
        text = 'RandomMiniBatchUtils parameters:\n'
        text += 'mini batch size %d\n' % self.mini_batch
        text += 'total batch size %d\n' % self.total_batch
        text += 'sub group names %s\n' % str(self.sub_group_names)
        for i, group_name in enumerate(self.sub_group_names):
            text += '%d) %s: count in mini batch: %s place in latent: %s\n' % (i, group_name, str(self.count_in_mini_bach_dict[group_name]), str(self.place_in_latent_dict[group_name]))
        _log.info(text)
        if self.debug:
            self.print_places_in_min_batch()

    def re_arrange_inject_noise(self, noises, group_name='id'):
        for place_in_mini_batch in self.place_in_mini_batch_dict[group_name]:
            for i in range(place_in_mini_batch[0], place_in_mini_batch[1], 2):
                for j in range(len(noises)):
                    noises[j][i+1, :, :, :] = noises[j][i, :, :, :].detach()
        return noises


if __name__ == '__main__':
    from gan_control.trainers.utils import mixing_noise, make_mini_batch_from_noise

    class DefualtArgs(object):
        def __init__(self, batch_size, mini_batch_size):
            self.batch = batch_size
            self.mini_batch = mini_batch_size

    batch_size = 8
    mini_batch_size = 8
    same_id = 2
    same_pose = 2
    pose_id_split = 2
    latent_size = 4
    args = DefualtArgs(batch_size, mini_batch_size)
    noise = mixing_noise(batch_size, latent_size, 1., 'cuda')
    print(noise)
    mini_noise_inputs = make_mini_batch_from_noise(noise, args)





