# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from torchvision import utils, transforms
import numpy as np
from tqdm import tqdm
import subprocess as sp, shlex

import face_alignment

from gan_control.utils.file_utils import read_json
from gan_control.utils.logging_utils import get_logger
from gan_control.utils.mini_batch_multi_split_utils import MiniBatchUtils
from gan_control.models.gan_model import Generator
from gan_control.evaluation.generation import gen_grid, gen_matrix, IterableModel
from gan_control.evaluation.age import calc_age_from_tensor_images
from gan_control.evaluation.orientation import calc_orientation_from_tensor_images
from gan_control.evaluation.expression import calc_expression_from_tensor_images, get_class
from gan_control.evaluation.hair import calc_hair_color_from_images
from gan_control.evaluation.face_alignment_utils.face_alignment_utils import align_tensor_images, load_lm3d

_log = get_logger(__name__)


class Inference():
    def __init__(self, model_dir):
        _log.info('Init inference class...')
        self.model_dir = model_dir
        self.model, self.batch_utils, self.config, self.ckpt_iter = self.retrieve_model(model_dir)
        self.noise = None
        self.noise = self.reset_noise()
        self.mean_w_latent = None
        self.mean_w_latents = None

    def calc_mean_w_latents(self):
        _log.info('Calc mean_w_latents...')
        mean_latent_w_list = []
        for i in range(100):
            latent_z = torch.randn(1000, self.config.model_config['latent_size'], device='cuda')
            if isinstance(self.model, torch.nn.DataParallel):
                latent_w = self.model.module.style(latent_z).cpu()
            else:
                latent_w = self.model.style(latent_z).cpu()
            mean_latent_w_list.append(latent_w.mean(dim=0).unsqueeze(0))
        self.mean_w_latent = torch.cat(mean_latent_w_list, dim=0).mean(0)
        self.mean_w_latents = {}
        for place_in_latent_key in self.batch_utils.place_in_latent_dict.keys():
            self.mean_w_latents[place_in_latent_key] = self.mean_w_latent[self.batch_utils.place_in_latent_dict[place_in_latent_key][0]: self.batch_utils.place_in_latent_dict[place_in_latent_key][1]]

    def reset_noise(self):
        if isinstance(self.model, torch.nn.DataParallel):
            self.noise = self.model.module.make_noise(device='cuda')
        else:
            self.noise = self.model.make_noise(device='cuda')

    @staticmethod
    def expend_noise(noise, batch_size):
        noise = [torch.cat([noise[n].clone() for _ in range(batch_size)], dim=0) for n in range(len(noise))]
        return noise

    @torch.no_grad()
    def gen_batch(self, batch_size=1, normalize=True, latent=None, input_is_latent=False, static_noise=True, truncation=1, **kwargs):
        if truncation < 1 and self.mean_w_latents is None:
            self.calc_mean_w_latents()
        injection_noise = None
        if latent is None:
            latent = torch.randn(batch_size, self.config.model_config['latent_size'], device='cuda')
        elif input_is_latent:
            latent = latent.cuda()
            for group_key in kwargs.keys():
                if group_key not in self.batch_utils.sub_group_names:
                    raise ValueError('group_key: %s not in sub_group_names %s' % (group_key, str(self.batch_utils.sub_group_names)))
                if isinstance(kwargs[group_key], str) and kwargs[group_key] == 'random':
                    if isinstance(self.model, torch.nn.DataParallel):
                        group_latent_w = self.model.style(torch.randn(latent.shape[0], self.config.model_config['latent_size'], device='cuda'))
                    else:
                        group_latent_w = self.model.module.style(torch.randn(latent.shape[0], self.config.model_config['latent_size'], device='cuda'))
                    group_latent_w = group_latent_w[:, self.batch_utils.place_in_latent_dict[group_key][0], self.batch_utils.place_in_latent_dict[group_key][0]]
                    latent[:, self.batch_utils.place_in_latent_dict[group_key][0], self.batch_utils.place_in_latent_dict[group_key][0]] = group_latent_w
        if static_noise:
            self.reset_noise()
            injection_noise = self.expend_noise(self.noise, latent.shape[0])

        if truncation < 1:
            if not input_is_latent:
                if isinstance(self.model, torch.nn.DataParallel):
                    latent = self.model.module.style(latent)
                else:
                    latent = self.model.style(latent)
                input_is_latent = True
            latent = self.calc_truncation(latent, truncation=truncation)

        tensor, latent_w = self.model([latent.cuda()], return_latents=True, input_is_latent=input_is_latent, noise=injection_noise)
        if normalize:
            tensor = tensor.mul(0.5).add(0.5).clamp(min=0., max=1.).cpu()
        return tensor, latent, latent_w

    def calc_truncation(self, latent_w, truncation=0.7):
        if truncation >= 1:
            return latent_w
        if self.mean_w_latents is None:
            self.calc_mean_w_latents()
        for key in self.batch_utils.place_in_latent_dict.keys():
            place_in_latent = self.batch_utils.place_in_latent_dict[key]
            latent_w[:, place_in_latent[0]: place_in_latent[1]] = \
                truncation * (latent_w[:, place_in_latent[0]: place_in_latent[1]] - torch.cat(
                    [self.mean_w_latents[key].clone().unsqueeze(0) for _ in range(latent_w.shape[0])], dim=0
                ).cuda()) + torch.cat(
                    [self.mean_w_latents[key].clone().unsqueeze(0) for _ in range(latent_w.shape[0])], dim=0
                ).cuda()
        return latent_w

    def generate_matrix_by_group(self, group, save_path=None, downsample=None):
        self.check_valid_group(group)
        same_chunk = self.batch_utils.place_in_latent_dict[group]
        image = gen_matrix(self.model, same_noise_per_id=False, downsample=downsample, same_chunk=same_chunk, same_noise_for_all=True)
        if save_path is not None:
            image_save_path = '%s.jpg' % save_path
            image.save(image_save_path)
            _log.info('Saved %s matrix to: %s' % (group, image_save_path))
        return image

    def interpolate_by_group(self, group, save_path,
                             batch=4,
                             num_of_intermediate_latents=4,
                             pics_per_interpolation=10,
                             interpolation='slerp',
                             same_noise=True,
                             downsample=None,
                             idx=None):
        self.check_valid_group(group)
        noise = self.model.make_noise()
        noise = [noise[i].expand(batch, -1, -1, -1).clone() for i in range(len(noise))]

        group_chunk = self.batch_utils.place_in_latent_dict[group]
        latent_1 = torch.randn(1, self.config.model_config['latent_size'], device='cuda')
        latent_1 = latent_1.expand(batch, -1)
        latent_2 = [torch.randn(batch, self.config.model_config['latent_size'], device='cuda') for _ in range(num_of_intermediate_latents)]
        sample_noise_1 = noise
        if same_noise:
            sample_noise_2 = [[noise[i].clone() for i in range(len(noise))] for _ in range(num_of_intermediate_latents)]
        else:
            sample_noise_2 = [self.model.module.make_noise(batch_size=batch) for _ in range(num_of_intermediate_latents)]
        z1_latent_start = latent_1[:,:group_chunk[0]]
        z1_latent_end = latent_1[:,group_chunk[1]:]
        z1_latent_group = latent_1[:,group_chunk[0]:group_chunk[1]]
        noise1 = sample_noise_1
        freeze_group_images = []
        freeze_not_group_images = []
        for z2, noise2 in tqdm(zip(latent_2, sample_noise_2)):
            for _, p in zip(range(pics_per_interpolation), np.linspace(0, 1, pics_per_interpolation)):
                interpolation_noise = [(1-p) * noise1[iii] + p * noise2[iii] for iii in range(len(noise1))]
                if interpolation == 'linear':
                    interpolation_latent_start = (1 - p) * z1_latent_start + p * z2[:,:group_chunk[0]]
                    interpolation_latent_end = (1 - p) * z1_latent_end + p * z2[:,group_chunk[1]:]
                    interpolation_latent_group = (1 - p) * z1_latent_group + p * z2[:,group_chunk[0]:group_chunk[1]]
                elif interpolation == 'slerp':
                    interpolation_latent_start = self.slerp(p, z1_latent_start, z2[:,:group_chunk[0]])
                    interpolation_latent_end = self.slerp(p, z1_latent_end, z2[:,group_chunk[1]:])
                    interpolation_latent_group = self.slerp(p, z1_latent_group, z2[:,group_chunk[0]:group_chunk[1]])

                else:
                    interpolation_latent_start = np.sqrt(1 - p) * z1_latent_start + np.sqrt(p) * z2[:,:group_chunk[0]]
                    interpolation_latent_end = np.sqrt(1 - p) * z1_latent_end + np.sqrt(p) * z2[:,group_chunk[1]:]
                    interpolation_latent_group = np.sqrt(1 - p) * z1_latent_group + np.sqrt(p) * z2[:,group_chunk[0]:group_chunk[1]]
                interpolation_latent_freeze_group = torch.cat([interpolation_latent_start, latent_1[:,group_chunk[0]:group_chunk[1]], interpolation_latent_end], dim=1)
                interpolation_latent_freeze_not_same_group = torch.cat([latent_1[:,:group_chunk[0]], interpolation_latent_group, latent_1[:,group_chunk[1]:]], dim=1)
                interpolation_image_freeze_group = self.gen_grid_image_from_latent(interpolation_latent_freeze_group, noise=interpolation_noise, nrow=batch, downsample=downsample)
                interpolation_image_freeze_not_same_group = self.gen_grid_image_from_latent(interpolation_latent_freeze_not_same_group, noise=interpolation_noise, nrow=batch, downsample=downsample)
                freeze_group_images.append(interpolation_image_freeze_group)
                freeze_not_group_images.append(interpolation_image_freeze_not_same_group)
            z1_latent_start = z2[:, :group_chunk[0]]
            z1_latent_end = z2[:, group_chunk[1]:]
            z1_latent_group = z2[:, group_chunk[0]:group_chunk[1]]
            noise1 = noise2
        freeze_group_images_path = os.path.join(save_path, 'freeze_group')
        freeze_not_group_images_path = os.path.join(save_path, 'freeze_not_same_group')
        os.makedirs(freeze_group_images_path, exist_ok=True)
        os.makedirs(freeze_not_group_images_path, exist_ok=True)
        freeze_group_gif_name = '../freeze_group_' + os.path.split(save_path)[-1] + ('' if idx is None else '%04d' % idx)
        freeze_not_group_gif_name = '../freeze_not_group_' + os.path.split(save_path)[-1] + ('' if idx is None else '%04d' % idx)
        self.make_interpolation_gif(freeze_group_images, freeze_group_images_path, name=freeze_group_gif_name)
        self.make_interpolation_gif(freeze_not_group_images, freeze_not_group_images_path, name=freeze_not_group_gif_name)

    def make_interpolation_gif(self, image_list, save_path, name='sample'):
        for i, image in tqdm(enumerate(image_list)):
            image.save(os.path.join(save_path, '%06d.jpg' % i))
        self.create_gif(save_path, name=name)

    @staticmethod
    def create_gif(save_dir, delay=50, name='sample'):
        sp.run(shlex.split(f'convert -delay {delay} -loop 0 -resize 1024x256 *.jpg {name}.gif'), cwd=save_dir)

    @staticmethod
    def slerp(val, low, high):
        low_norm = low / torch.norm(low, dim=1, keepdim=True)
        high_norm = high / torch.norm(high, dim=1, keepdim=True)
        omega = torch.acos((low_norm * high_norm).sum(1))
        so = torch.sin(omega)
        res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
        return res

    def gen_grid_image_from_latent(self, latent, noise=None, nrow=4, downsample=None):
        with torch.no_grad():
            sample, _ = self.model([latent.cuda()], noise=noise)
        sample = sample.mul(0.5).add(0.5).clamp(min=0., max=1.).cpu()
        grid_image = transforms.ToPILImage()(utils.make_grid(sample, nrow=nrow))
        if downsample is not None:
            grid_image = transforms.Resize((grid_image.size[0] // downsample, grid_image.size[1] // downsample))(grid_image)
        return grid_image

    def check_valid_group(self, group):
        if group not in self.batch_utils.sub_group_names:
            raise ValueError(
                'group: %s not in valid group names for this model\n'
                'Valid group names are:\n'
                '%s' % str(self.batch_utils.sub_group_names)
            )

    @staticmethod
    def retrieve_model(model_dir):
        config_path = os.path.join(model_dir, 'args.json')

        _log.info('Retrieve config from %s' % config_path)
        checkpoints_path = os.path.join(model_dir, 'checkpoint')
        ckpt_list = list(os.listdir(checkpoints_path))
        ckpt_list.sort()
        ckpt_path = ckpt_list[-1]
        ckpt_iter = ckpt_path.split('.')[0]
        config = read_json(config_path, return_obj=True)
        ckpt = torch.load(os.path.join(checkpoints_path, ckpt_path))

        batch_utils = None
        if not config.model_config['vanilla']:
            _log.info('Init Batch Utils...')
            batch_utils = MiniBatchUtils(
                config.training_config['mini_batch'],
                config.training_config['sub_groups_dict'],
                total_batch=config.training_config['batch']
            )
            batch_utils.print()

        _log.info('Init Model...')
        model = Generator(
            config.model_config['size'],
            config.model_config['latent_size'],
            config.model_config['n_mlp'],
            channel_multiplier=config.model_config['channel_multiplier'],
            out_channels=config.model_config['img_channels'],
            split_fc=config.model_config['split_fc'],
            fc_config=None if config.model_config['vanilla'] else batch_utils.get_fc_config(),
            conv_transpose=config.model_config['conv_transpose'],
            noise_mode=config.model_config['g_noise_mode']
        ).cuda()
        _log.info('Loading Model: %s, ckpt iter %s' % (model_dir, ckpt_iter))
        model.load_state_dict(ckpt['g_ema'])
        model = torch.nn.DataParallel(model)
        model.eval()

        return model, batch_utils, config, ckpt_iter

    @staticmethod
    def retrieve_trainer(model_dir, default_trainer=True):
        from gan_control.trainers.generator_trainer import GeneratorTrainer

        config_path = os.path.join(model_dir, 'args.json')
        if default_trainer:
            config_path = os.path.join('path to default trainer', 'args.json')
        trainer = GeneratorTrainer(config_path, init_dirs=False)
        return trainer

    def extract_controls_from_images(self, images):
        if images.min() > -0.001 or images.max() > 1.0001:
            _log.warning('images pixel range is min=%.2f, max=%.2f but should be in range [-1, 1]' % (images.min(), images.max()))
        if not hasattr(self, 'trainer'):
            _log.info('Init trainer for attribute models, this is a one time init')
            self.trainer = self.retrieve_trainer(self.model_dir)
        lm3D = load_lm3d()
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
        self.trainer.recon_3d_loss_class.skeleton_model.module.config['center_crop'] = None

        age = calc_age_from_tensor_images(self.trainer.age_class, images)
        orientation = calc_orientation_from_tensor_images(self.trainer.pose_orientation_class, images)
        expression_q = calc_expression_from_tensor_images(self.trainer.pose_expression_class, images)
        hair_color = calc_hair_color_from_images(self.trainer.hair_loss_class, images)
        recon_3d_features = self.trainer.recon_3d_loss_class.calc_features(align_tensor_images(images, fa=fa, lm3D=lm3D))
        id_futures, ex_futures, tex_futures, angles_futures, gamma_futures, xy_futures, z_futures = self.trainer.recon_3d_loss_class.skeleton_model.module.extract_futures_from_vec(recon_3d_features)
        orientation = torch.cat(orientation, dim=0)
        gamma3d = gamma_futures[-1]
        expression3d = ex_futures[-1]
        orientation3d = angles_futures[-1]
        controls_dict = {
            'age': age.cpu(),
            'orientation': orientation.cpu(),
            'expression_q': expression_q,
            'hair': hair_color.cpu(),
            'gamma3d': gamma3d.cpu(),
            'expression3d': expression3d.cpu(),
            'orientation3d': orientation3d.cpu(),
        }
        return controls_dict

