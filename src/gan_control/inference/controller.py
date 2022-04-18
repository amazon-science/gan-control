# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch

from gan_control.utils.file_utils import read_json
from gan_control.utils.logging_utils import get_logger
from gan_control.inference.inference import Inference
from gan_control.models.controller_model import FcStack

_log = get_logger(__name__)


class Controller(Inference):
    def __init__(self, controller_dir):
        _log.info('Init Controller class...')
        super(Controller, self).__init__(os.path.join(controller_dir, 'generator'))
        self.fc_controls = {}
        self.config_controls = {}
        for sub_group_name in self.batch_utils.sub_group_names:
            controller, controller_config = self.retrieve_controller(controller_dir, sub_group_name)
            self.fc_controls[sub_group_name] = controller
            self.config_controls[sub_group_name] = controller_config
        controller, controller_config = self.retrieve_controller(controller_dir, 'expression_q')
        self.fc_controls['expression_q'] = controller
        self.config_controls['expression_q'] = controller_config

    @torch.no_grad()
    def gen_batch_by_controls(self, batch_size=1, latent=None, normalize=True, input_is_latent=False, static_noise=True, **kwargs):
        if latent is None:
            latent = torch.randn(batch_size, self.config.model_config['latent_size'], device='cuda')
        latent = latent.clone()
        if input_is_latent:
            latent_w = latent
        else:
            if isinstance(self.model, torch.nn.DataParallel):
                latent_w = self.model.module.style(latent.cuda())
            else:
                latent_w = self.model.style(latent.cuda())
        for group_key in kwargs.keys():
            if self.check_if_group_has_control(group_key):
                if group_key == 'expression' and kwargs[group_key].shape[1] == 8:
                    group_w_latent = self.fc_controls['expression_q'](kwargs[group_key].cuda().float())
                else:
                    group_w_latent = self.fc_controls[group_key](kwargs[group_key].cuda().float())
                latent_w = self.insert_group_w_latent(latent_w, group_w_latent, group_key)
        injection_noise = None
        if static_noise:
            injection_noise = self.expend_noise(self.noise, latent.shape[0])
        tensor, _ = self.model([latent_w], input_is_latent=True, noise=injection_noise)
        if normalize:
            tensor = tensor.mul(0.5).add(0.5).clamp(min=0., max=1.).cpu()
        return tensor, latent, latent_w

    def generate_group_w_latent(self, group_key: str, value: torch.Tensor):
        group_w_latent = self.fc_controls[group_key](value)
        return group_w_latent

    def insert_group_w_latent(self, latent_w, group_w_latent, group):
        if group_w_latent.ndim != group_w_latent.ndim:
            raise ValueError(f'group_w_latent.ndim ({group_w_latent.ndim}) must equal latent_w.ndim ({latent_w.ndim})')
        latent_per_resolution = latent_w.ndim == 3
        if latent_per_resolution:
            latent_w[:, :,
            self.batch_utils.place_in_latent_dict[group][0]:self.batch_utils.place_in_latent_dict[group][1]] = \
                group_w_latent
        else:
            latent_w[:, self.batch_utils.place_in_latent_dict[group][0]:self.batch_utils.place_in_latent_dict[group][1]] = group_w_latent

        return latent_w

    def get_group_w_latent(self, latent_w: torch.Tensor, group: str):
        latent_per_resolution = latent_w.ndim == 3
        if latent_per_resolution:
            return latent_w[:, :,
                   self.batch_utils.place_in_latent_dict[group][0]:self.batch_utils.place_in_latent_dict[group][1]]
        else:
            return latent_w[:,
                   self.batch_utils.place_in_latent_dict[group][0]:self.batch_utils.place_in_latent_dict[group][1]]

    @staticmethod
    def get_controller_dir(controller_dir, sub_group_name):
        str_length = len(sub_group_name)
        possible_dirs = list(os.listdir(controller_dir))
        for possible_dir in possible_dirs:
            if len(possible_dir) >= str_length and sub_group_name == possible_dir[:str_length] and not (sub_group_name == 'expression' and possible_dir.startswith('expression_q')):
                return os.path.join(controller_dir, possible_dir)
        return None

    def retrieve_controller(self, controller_dir, sub_group_name):
        controller_dir_path = self.get_controller_dir(controller_dir, sub_group_name)
        if controller_dir_path is None:
            _log.info('No %s controller' % sub_group_name)
            return None, None
        config_path = os.path.join(controller_dir_path, 'args.json')
        _log.info('Retrieve controller config from %s' % config_path)
        checkpoints_path = os.path.join(controller_dir_path, 'checkpoint')
        ckpt_list = list(os.listdir(checkpoints_path))
        ckpt_list.sort()
        ckpt_path = ckpt_list[-1]
        ckpt_iter = ckpt_path.split('.')[0]
        config = read_json(config_path, return_obj=True)
        ckpt = torch.load(os.path.join(checkpoints_path, ckpt_path))
        group_chunk = self.batch_utils.place_in_latent_dict[sub_group_name if sub_group_name is not 'expression_q' else 'expression']
        group_latent_size = group_chunk[1] - group_chunk[0]

        _log.info('Init %s Controller...' % sub_group_name)
        controller = FcStack(config.model_config['lr_mlp'], config.model_config['n_mlp'], config.model_config['in_dim'], config.model_config['mid_dim'], group_latent_size).cuda()
        controller.print()

        _log.info('Loading Controller: %s, ckpt iter %s' % (controller_dir_path, ckpt_iter))
        controller.load_state_dict(ckpt['controller'])
        controller.eval()

        return controller, config,

    def check_if_group_has_control(self, group):
        if group not in self.fc_controls.keys():
            raise ValueError('group: %s has no control' % group)
        return True
