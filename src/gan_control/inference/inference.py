# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from torchvision import utils, transforms

from gan_control.utils.file_utils import read_json
from gan_control.utils.logging_utils import get_logger
from gan_control.utils.mini_batch_multi_split_utils import MiniBatchUtils
from gan_control.models.gan_model import Generator


_log = get_logger(__name__)


class Inference():
    def __init__(self, model_dir, device='cuda:0'):
        _log.info('Init inference class...')
        self.model_dir = model_dir
        self.device = device
        self.model, self.batch_utils, self.config, self.ckpt_iter = self.retrieve_model(model_dir, device)
        self.noise = None
        self.reset_noise()
        self.mean_w_latent = None
        self.mean_w_latents = None

    def calc_mean_w_latents(self):
        _log.info('Calc mean_w_latents...')
        mean_latent_w_list = []
        for i in range(100):
            latent_z = torch.randn(1000, self.config.model_config['latent_size'], device=self.device)
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
            self.noise = self.model.module.make_noise(device=self.device)
        else:
            self.noise = self.model.make_noise(device=self.device)

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
            latent = torch.randn(batch_size, self.config.model_config['latent_size'], device=self.device)
        elif input_is_latent:
            latent = latent.to(self.device)
            for group_key in kwargs.keys():
                if group_key not in self.batch_utils.sub_group_names:
                    raise ValueError('group_key: %s not in sub_group_names %s' % (group_key, str(self.batch_utils.sub_group_names)))
                if isinstance(kwargs[group_key], str) and kwargs[group_key] == 'random':
                    group_latent_w = self.model.style(torch.randn(latent.shape[0], self.config.model_config['latent_size'], device=self.device))
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
            for key in self.batch_utils.place_in_latent_dict.keys():
                place_in_latent = self.batch_utils.place_in_latent_dict[key]
                latent[:, place_in_latent[0]: place_in_latent[1]] = \
                    truncation * (latent[:, place_in_latent[0]: place_in_latent[1]] - torch.cat(
                        [self.mean_w_latents[key].clone().unsqueeze(0) for _ in range(latent.shape[0])], dim=0
                    ).to(self.device)) + torch.cat(
                        [self.mean_w_latents[key].clone().unsqueeze(0) for _ in range(latent.shape[0])], dim=0
                    ).to(self.device)

        tensor, latent_w = self.model([latent.to(self.device)], return_latents=True, input_is_latent=input_is_latent, noise=injection_noise)
        if normalize:
            tensor = tensor.mul(0.5).add(0.5).clamp(min=0., max=1.).cpu()
        return tensor, latent, latent_w

    def check_valid_group(self, group):
        if group not in self.batch_utils.sub_group_names:
            raise ValueError(
                'group: %s not in valid group names for this model\n'
                'Valid group names are:\n'
                '%s' % (group, str(self.batch_utils.sub_group_names))
            )

    @staticmethod
    def make_resized_grid_image(image_tensors, resize=None, nrow=8):
        grid_image = transforms.ToPILImage()(utils.make_grid(image_tensors, nrow=nrow))
        if resize is not None:
            grid_image = transforms.Resize(resize)(grid_image)
        return grid_image

    @staticmethod
    def retrieve_model(model_dir, device):
        config_path = os.path.join(model_dir, 'args.json')

        _log.info('Retrieve config from %s' % config_path)
        checkpoints_path = os.path.join(model_dir, 'checkpoint')
        ckpt_list = list(os.listdir(checkpoints_path))
        ckpt_list.sort()
        ckpt_path = ckpt_list[-1]
        ckpt_iter = ckpt_path.split('.')[0]
        config = read_json(config_path, return_obj=True)
        ckpt = torch.load(os.path.join(checkpoints_path, ckpt_path), map_location=device)

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
        ).to(device)
        _log.info('Loading Model: %s, ckpt iter %s' % (model_dir, ckpt_iter))
        model.load_state_dict(ckpt['g_ema'])
        model = torch.nn.DataParallel(model)
        model.eval()

        return model, batch_utils, config, ckpt_iter


