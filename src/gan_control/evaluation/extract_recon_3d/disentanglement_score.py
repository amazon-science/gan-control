# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from tqdm import tqdm
import pandas as pd
import numpy as np

from igt_res_gan.evaluation.extract_recon_3d.disentanglement_dataloader import get_disentanglement_data_loader
from igt_res_gan.evaluation.extract_recon_3d.extract_recon_3d import calc_vectors_mean_and_std
from igt_res_gan.utils.file_utils import read_json
from igt_res_gan.losses.loss_model import LossModelClass
from igt_res_gan.utils.logging_utils import get_logger

_log = get_logger(__name__)


def key_to_std_mapping(key):
    mapping_dict = {'u_id': 'id_std', 'u_exp': 'ex_std', 'u_gamma': 'gamma_std', 'u_pose': 'angles_std'}
    return mapping_dict[key]


def calc_ds_i(key, u_dataloader, ffhq_std_df=None, embedding_loss_model=None, recon_3d_loss_class=None, debug=True):
    _log.info('Calc DS for Ui:%s' % (key))
    columns = ['uj', 'ui', 'id', 'id3d', 'ex', 'angles', 'gamma', 'ui_name']
    properties_df = pd.DataFrame(columns=columns)
    pbar = tqdm(u_dataloader)
    for iter, (batch, uj, ui) in enumerate(pbar):
        if debug and iter > 25:
            break
        tensor_images = batch.cuda()
        #uj = batch[1]
        #ui = batch[2]
        id_embedding_perceptual_features = embedding_loss_model.calc_features(tensor_images)
        pose_features = recon_3d_loss_class.calc_features(tensor_images)
        id_futures, ex_futures, tex_futures, angles_futures, gamma_futures, xy_futures, z_futures = recon_3d_loss_class.skeleton_model.module.extract_futures_from_vec(pose_features)
        for i in range(id_futures[0].shape[0]):
            inter_dict = {'id': id_embedding_perceptual_features[-1][i].cpu().numpy(),
                          'id3d': id_futures[-1][i].cpu().numpy(),
                          'ex': ex_futures[-1][i].cpu().numpy(),
                          'angles': angles_futures[-1][i].cpu().numpy(),
                          'gamma': gamma_futures[-1][i].cpu().numpy(),
                          'uj': uj[i].item(),
                          'ui': ui[i].item(),
                          'ui_name': key}
            properties_df = properties_df.append(inter_dict, ignore_index=True)
            properties_df = properties_df.reset_index(drop=True)
    pbar.close()
    std_df = pd.DataFrame(columns=['id_std', 'id3d_std', 'ex_std', 'angles_std', 'gamma_std'])
    for uj_num in range(len(np.unique(properties_df.uj.to_numpy()))):
        sub_properties_df = properties_df.loc[properties_df.uj == uj_num]
        _, id_std = calc_vectors_mean_and_std(sub_properties_df.id.to_list())
        _, id3d_std = calc_vectors_mean_and_std(sub_properties_df.id3d.to_list())
        _, ex_std = calc_vectors_mean_and_std(sub_properties_df.ex.to_list())
        _, angles_std = calc_vectors_mean_and_std(sub_properties_df.angles.to_list())
        _, gamma_std = calc_vectors_mean_and_std(sub_properties_df.gamma.to_list())
        inter_dict = {'id_std': id_std,
                      'id3d_std': id3d_std,
                      'ex_std': ex_std,
                      'angles_std': angles_std,
                      'gamma_std': gamma_std}
        std_df = std_df.append(inter_dict, ignore_index=True)
        std_df = std_df.reset_index(drop=True)
    sigmas = {}
    for std_key in ['id_std', 'ex_std', 'angles_std', 'gamma_std']:
        sigmas[std_key] = std_df[std_key].to_numpy().mean() / ffhq_std_df[std_key].to_list()[0]
    sigma_i = sigmas[key_to_std_mapping(key)]
    _log.info('Sigma Ui (%s): %s' % (key, sigma_i))
    _log.info('Sigmas: %s' % str(sigmas))

    ds_i = 1
    for std_key in ['id_std', 'ex_std', 'angles_std', 'gamma_std']:
        if std_key == key_to_std_mapping(key):
            continue
        ds_i = ds_i * sigma_i / sigmas[std_key]
    _log.info('DSi (%s): %s' % (key, ds_i))
    return ds_i


if __name__ == '__main__':
    import argparse
    #--datasets_path /mnt/md4/orville/Alon/research_desk/ds_outputs/age015id025exp02hai04ori02gam15_normal_20200913-121433 --save_path /mnt/md4/orville/Alon/research_desk/ds_outputs/age015id025exp02hai04ori02gam15_normal_20200913-121433/ds_df.csv
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='../../configs/configs512/default.json')
    parser.add_argument('--datasets_path', type=str, default='/mnt/md4/orville/Alon/research_desk/microsoft_discogan')
    parser.add_argument('--ffhq_std_path', type=str, default='/mnt/md4/orville/Alon/research_desk/dataframs/ffhq_extract_recon_3d/std_df.csv')
    parser.add_argument('--save_path', type=str, default='/mnt/md4/orville/Alon/research_desk/microsoft_discogan/ds_df.csv')
    parser.add_argument('--batch_size', type=int, default=50)

    args = parser.parse_args()
    config = read_json(args.config_path, return_obj=True)
    recon_3d_loss_class = LossModelClass(config.training_config['recon_3d_loss'], loss_name='recon_3d_loss', mini_batch_size=config.training_config['mini_batch'])
    embedding_loss_model = LossModelClass(config.training_config['embedding_loss'], loss_name='embedding_loss', mini_batch_size=config.training_config['mini_batch'], device="cuda")
    u_id_dataloader = get_disentanglement_data_loader({'path': os.path.join(args.datasets_path, 'u_id'), 'workers': 32}, batch_size=args.batch_size)
    u_exp_dataloader = get_disentanglement_data_loader({'path': os.path.join(args.datasets_path, 'u_exp'), 'workers': 32}, batch_size=args.batch_size)
    u_gamma_dataloader = get_disentanglement_data_loader({'path': os.path.join(args.datasets_path, 'u_gamma'), 'workers': 32}, batch_size=args.batch_size)
    u_pose_dataloader = get_disentanglement_data_loader({'path': os.path.join(args.datasets_path, 'u_pose'), 'workers': 32}, batch_size=args.batch_size)
    u_dataloader_dict = {'u_id': u_id_dataloader, 'u_exp': u_exp_dataloader, 'u_gamma': u_gamma_dataloader, 'u_pose': u_pose_dataloader}
    ffhq_std_df = pd.read_csv(args.ffhq_std_path)
    ds_i_dict = {}
    for key in u_dataloader_dict.keys():
        ds_i_dict[key] = calc_ds_i(key, u_dataloader_dict[key], ffhq_std_df=ffhq_std_df, embedding_loss_model=embedding_loss_model, recon_3d_loss_class=recon_3d_loss_class)
    pd.DataFrame(ds_i_dict, index=[0]).to_csv(args.save_path)