# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from gan_control.utils.logging_utils import get_logger

_log = get_logger(__name__)


class MiniBatchUtils():
    def __init__(self, batch_size, same_id, same_pose, pose_id_split, total_batch=8):
        self.local_batch_size = batch_size
        self.normal = batch_size - same_id - same_pose
        self.same_id = same_id
        self.same_pose = same_pose
        self.pose_id_split = pose_id_split
        self.num_of_mini_batchs = total_batch // batch_size

    def get_num_of_mini_batches_with_normals(self, batch_num):
        if self.normal !=0:
            return self.num_of_mini_batchs
        else:
            return 0

    def get_num_of_mini_batches_with_same_pose(self, batch_num):
        if self.same_pose !=0:
            return self.num_of_mini_batchs
        else:
            return 0

    def get_num_of_mini_batches_with_same_id(self, batch_num):
        if self.same_id !=0:
            return self.num_of_mini_batchs
        else:
            return 0


    def get_normals(self, batch, batch_num):
        if self.normal == 0:
            return None
        return batch[0:self.normal]

    def get_same_ids(self, batch, batch_num):
        if self.same_id == 0:
            return None
        return batch[self.normal:self.normal + self.same_id]

    def get_same_poses(self, batch, batch_num):
        if self.same_pose == 0:
            return None
        return batch[self.normal + self.same_id:]

    def sanity_check(self, z_batch):
        flag_str = None
        print('\n#########Sanity Check#########')
        for i in range(0, z_batch[0].shape[0], 2):
            for j in range(len(z_batch)):
                same_id = torch.sum(z_batch[j][i, self.pose_id_split:] - z_batch[j][i + 1, self.pose_id_split:]).cpu().item() == 0
                same_pose = torch.sum(z_batch[j][i, :self.pose_id_split] - z_batch[j][i + 1, :self.pose_id_split:]).cpu().item() == 0
                if i+1 <= self.normal:
                    good = (not same_id) and (not same_pose)
                elif i+1 <= self.normal + self.same_id:
                    good = same_id and (not same_pose)
                elif i+1 <= self.normal + self.same_id + self.same_pose:
                    good = (not same_id) and same_pose
                print_str = 'batch %02d and %02d: id %s, pose %s, OK %s' % (i, i+1, 'same' if same_id else 'diff', 'same' if same_pose else 'diff', 'yes' if good else '*********NOOOOO********')
                if not good:
                    flag_str = print_str
                good = 'yes' if good else '*********NOOOOO********'
                print(print_str)
        print('###############################')
        if flag_str is not None:
            raise Exception(flag_str)

    # latent_id_pose_split
    def re_arrange_z(self, z_batch, batch_num):
        # match ids
        for i in range(self.normal, self.normal + self.same_id, 2):
            z_batch[0][i + 1, self.pose_id_split:] = z_batch[0][i, self.pose_id_split:].detach()
        # match poses
        for i in range(self.normal + self.same_id, self.normal + self.same_id + self.same_pose, 2):
            z_batch[0][i + 1, 0:self.pose_id_split] = z_batch[0][i, 0:self.pose_id_split].detach()
        if len(z_batch) > 1:
            for i in range(1, len(z_batch)):
                z_batch[i][self.normal:, :] = z_batch[0][self.normal:, :]
        return z_batch

    def extract_ids_poses_normals_from_list(self, feature_list, batch_num):
        same_id_list = []
        same_pose_list = []
        normal_list = []
        for i in range(len(feature_list)):
            same_id_list.append(self.get_same_ids(feature_list[i], None))
            same_pose_list.append(self.get_same_poses(feature_list[i], None))
            normal_list.append(self.get_normals(feature_list[i], None))
        return same_id_list, same_pose_list, normal_list

    def print(self, intro=True):
        if intro:
            print('\n#####Using old#####')
        print('Mini batch size: %d, normals: %d, same ids: %d, same poses: %d' % (self.local_batch_size, self.normal, self.same_id, self.same_pose))

    def re_arrange_inject_noise(self, noises, batch_num):
        for i in range(self.normal, self.normal + self.same_id, 2):
            for j in range(len(noises)):
                noises[j][i+1,:,:,:] = noises[j][i,:,:,:].detach()
        return noises


class MiniBatchUtilsV2():
    def __init__(self, mini_batch_size, same_id, same_pose, pose_id_split, total_batch=8):
        self.batch_size = total_batch
        self.mini_batch_size = mini_batch_size
        self.total_normal = self.batch_size - same_id - same_pose
        self.total_same_id = same_id
        self.total_same_pose = same_pose
        self.pose_id_split = pose_id_split
        self.num_of_mini_batches = self.batch_size // mini_batch_size
        normal_sum = self.total_normal
        same_id_sum = self.total_same_id
        same_pose_sum = self.total_same_pose
        self.mini_batch_list = []
        self.num_of_normal_mini_batches = 0
        self.num_of_same_id_mini_batches = 0
        self.num_of_same_pose_mini_batches = 0

        for i in range(self.num_of_mini_batches):
            batch_left = self.mini_batch_size
            if normal_sum > 0:
                d_normal = batch_left if normal_sum - batch_left > 0 else normal_sum
                batch_left -= d_normal
                normal_sum -= d_normal
            else:
                d_normal = 0
            if same_id_sum > 0 and batch_left > 0:
                d_same_id = batch_left if same_id_sum - batch_left > 0 else same_id_sum
                batch_left -= d_same_id
                same_id_sum -= d_same_id
            else:
                d_same_id = 0
            if same_pose_sum > 0 and batch_left > 0:
                d_same_pose = batch_left if same_pose_sum - batch_left > 0 else same_pose_sum
                batch_left -= d_same_pose
                same_pose_sum -= d_same_pose
            else:
                d_same_pose = 0
            self.mini_batch_list.append(MiniBatchUtils(mini_batch_size, d_same_id, d_same_pose, pose_id_split, total_batch=self.batch_size))

            if d_normal > 0:
                self.num_of_normal_mini_batches += 1
            if d_same_id > 0:
                self.num_of_same_id_mini_batches += 1
            if d_same_pose > 0:
                self.num_of_same_pose_mini_batches += 1

    def get_normals(self, batch, batch_num):
        return self.mini_batch_list[batch_num].get_normals(batch, None)

    def get_same_ids(self, batch, batch_num):
        return self.mini_batch_list[batch_num].get_same_ids(batch, None)

    def get_same_poses(self, batch, batch_num):
        return self.mini_batch_list[batch_num].get_same_poses(batch, None)

    def full_sanity_check(self, z_batch):
        flag_str = None
        print('\n#########Sanity Check#########')
        for mini_batch_num in range(self.num_of_mini_batches):
            new_flag_str = self.mini_batch_list[mini_batch_num].sanity_check(self.mini_batch_list[mini_batch_num].re_arrange_z(z_batch, None))
            if flag_str is not None:
                flag_str = new_flag_str
        print('###############################')
        if flag_str is not None:
            raise Exception(flag_str)

    def sanity_check(self, z_batch, batch_num):
        print('\n#########Sanity Check: mini batch:%d#########' % batch_num)
        flag_str = self.mini_batch_list[batch_num].sanity_check(z_batch)
        print('##############################################')
        if flag_str is not None:
            raise Exception(flag_str)

    # latent_id_pose_split
    def re_arrange_z(self, z_batch, batch_num):
        return self.mini_batch_list[batch_num].re_arrange_z(z_batch, None)

    def extract_ids_poses_normals_from_list(self, feature_list, batch_num):
        return self.mini_batch_list[batch_num].extract_ids_poses_normals_from_list(feature_list, None)

    def get_num_of_mini_batches_with_normals(self, batch_num):
        return self.num_of_normal_mini_batches

    def get_num_of_mini_batches_with_same_pose(self, batch_num):
        return self.num_of_same_pose_mini_batches

    def get_num_of_mini_batches_with_same_id(self, batch_num):
        return self.num_of_same_id_mini_batches

    def print(self, intro=True):
        if intro:
            print('\n#####Using batch utils v2#####')
        for batch_num in range(len(self.mini_batch_list)):
            self.mini_batch_list[batch_num].print(intro=False)
        print('num of same id mini batches: %d' % self.get_num_of_mini_batches_with_same_id(None))
        print('num of same pose mini batches: %d' % self.get_num_of_mini_batches_with_same_pose(None))
        print('num of normal mini batches: %d' % self.get_num_of_mini_batches_with_normals(None))

    def re_arrange_inject_noise(self, noises, batch_num):
        return self.mini_batch_list[batch_num].re_arrange_inject_noise(noises, None)


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

    mini_batch_utils_v2 = MiniBatchUtilsV2(mini_batch_size, same_id, same_pose, pose_id_split, total_batch=batch_size)
    for batch_num, mini_noise in enumerate(mini_noise_inputs):
        print('old noise')
        print(mini_noise)
        print(mini_batch_utils_v2.mini_batch_list[batch_num].normal)
        print(mini_batch_utils_v2.total_normal)
        new_noise = mini_batch_utils_v2.re_arrange_z(mini_noise, batch_num)
        print('new noise')
        print(new_noise)



