# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import json
import time
import math

from tqdm import tqdm
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils

from gan_control.utils.file_utils import read_json, setup_logging_from_args
# from igt_res_gan.utils.mini_batch_utils import MiniBatchUtils
from gan_control.utils.mini_batch_multi_split_utils import MiniBatchUtils
from gan_control.utils.mini_batch_random_multi_split_utils import RandomMiniBatchUtils
from gan_control.fid_utils.calc_inception import load_patched_inception_v3
from gan_control.trainers.utils import accumulate, make_noise, mixing_noise, requires_grad, make_mini_batch_from_noise, set_grad_none
from gan_control.models.gan_model import Generator, Discriminator
from gan_control.losses.loss_model import LossModelClass
from gan_control.evaluation.tracker import Tracker
from gan_control.utils.logging_utils import get_logger


from gan_control.trainers.non_leaking import augment

_log = get_logger(__name__)


class GeneratorTrainer():
    def __init__(self, config_path, init_dirs=True):
        _log.info('Init Trainer...')
        self.device = "cuda"
        self.init_config(config_path, init_dirs=init_dirs)
        if init_dirs:
            self.init_dirs()
        self.init_data_set()
        self.init_losses()
        self.init_tensorboard(init_dirs)
        self.batch_utils = None
        if not self.model_config['vanilla']:
            _log.info('Init Batch Utils...')
        else:
            _log.warning('Init Batch Utils... (Batch Utils will not be used for training only for evaluation)')
        if self.training_config['mini_batch_mode'] == 'random':
            self.batch_utils = RandomMiniBatchUtils(
                self.training_config['mini_batch'],
                self.training_config['sub_groups_dict'],
                total_batch=self.training_config['batch'],
                debug=self.training_config['debug']
            )
        else:
            self.batch_utils = MiniBatchUtils(
                self.training_config['mini_batch'],
                self.training_config['sub_groups_dict'],
                total_batch=self.training_config['batch'],
                debug=self.training_config['debug']
            )
        self.batch_utils.print()
        self.init_models_and_optim()
        self.init_evaluation()
        self.tracker = Tracker(
            torch.randn(4, 512).cpu(),
            self.generator.module.make_noise(batch_size=4, device="cpu"),
            self.writer,
            self.inception,
            self.model_config['g_noise_mode'],
            fid_config=self.evaluation_config['fid'],
            separability_config=self.evaluation_config['separability'],
            orientation_hist_config=self.evaluation_config['orientation_hist'],
            expression_bar_config=self.evaluation_config['expression_bar']
        )
        _log.info('Trainer initiated')

    def init_config(self, config_path, init_dirs=True):
        self.config = read_json(config_path, return_obj=True)
        if hasattr(self.config, 'add_weight_to_name') and self.config.add_weight_to_name and not self.config.model_config['vanilla']:
            self.config.save_name = self.add_weight_to_name(self.config.save_name)
        if self.config.training_config['debug']:
            self.config.save_name = self.config.save_name + '_debug'
            self.config.results_dir = self.config.results_dir + '_debug'
        if init_dirs:
            self.config.save_dir = setup_logging_from_args(self.config)
        self.training_config = self.config.training_config
        self.tensorboard_config = self.config.tensorboard_config
        self.model_config = self.config.model_config
        self.data_config = self.config.data_config
        self.ckpt_config = self.config.ckpt_config
        self.monitor_config = self.config.monitor_config
        self.evaluation_config = self.config.evaluation_config
        self.config_checks()

    def config_checks(self):
        _log.info('Checking config...')
        if self.training_config.__contains__('augment') and self.training_config['augment']['enabled'] and self.training_config['mini_batch'] != self.training_config['batch']:
            raise ValueError('augment does not support mini_batch != batch')
        if self.training_config.__contains__('marge_fc') and self.model_config['marge_fc'] and self.model_config['split_fc']:
            raise ValueError('marge_fc and split_fc are true')
        if self.training_config.__contains__('mini_batch_mode') and self.training_config['mini_batch_mode'] not in ['normal', 'random']:
            raise ValueError('mini_batch_mode: %s not in [normal, random]' % self.training_config['mini_batch_mode'])
        # if list(self.training_config['sub_groups_dict'].keys())
        _log.warning('TODO: implement config checks')
        _log.info('Config passed checking')

    def init_dirs(self):
        os.makedirs(os.path.join(self.config.save_dir, 'checkpoint'), exist_ok=True)
        os.makedirs(os.path.join(self.config.save_dir, 'graphs'), exist_ok=True)
        os.makedirs(os.path.join(self.config.save_dir, 'buckets'), exist_ok=True)
        os.makedirs(os.path.join(self.config.save_dir, 'images', 'sample'), exist_ok=True)
        os.makedirs(os.path.join(self.config.save_dir, 'images', 'matrix'), exist_ok=True)
        os.makedirs(os.path.join(self.config.save_dir, 'images', 'orientation_matrix'), exist_ok=True)
        os.makedirs(os.path.join(self.config.save_dir, 'images', 'expression_matrix'), exist_ok=True)
        os.makedirs(os.path.join(self.config.save_dir, 'images', 'attribute_matrix'), exist_ok=True)
        os.makedirs(os.path.join(self.config.save_dir, 'images', 'other_matrix'), exist_ok=True)
        os.makedirs(os.path.join(self.config.save_dir, 'images', 'age_matrix'), exist_ok=True)
        os.makedirs(os.path.join(self.config.save_dir, 'images', 'hair_matrix'), exist_ok=True)

    def init_models_and_optim(self):
        self.generator = Generator(
            self.model_config['size'],
            self.model_config['latent_size'],
            self.model_config['n_mlp'],
            channel_multiplier=self.model_config['channel_multiplier'],
            out_channels=self.model_config['img_channels'],
            split_fc=self.model_config['split_fc'],
            marge_fc=self.model_config['marge_fc'],
            fc_config=self.batch_utils.get_fc_config(),
            conv_transpose=self.model_config['conv_transpose'],
            noise_mode=self.model_config['g_noise_mode']
        ).to(self.device)
        self.g_ema = Generator(
            self.model_config['size'],
            self.model_config['latent_size'],
            self.model_config['n_mlp'],
            channel_multiplier=self.model_config['channel_multiplier'],
            out_channels=self.model_config['img_channels'],
            split_fc=self.model_config['split_fc'],
            marge_fc=self.model_config['marge_fc'],
            fc_config=self.batch_utils.get_fc_config(),
            conv_transpose=self.model_config['conv_transpose'],
            noise_mode=self.model_config['g_noise_mode']
        ).to(self.device)
        self.discriminator = Discriminator(
            self.model_config['size'],
            channel_multiplier=self.model_config['channel_multiplier'],
            in_channels=self.model_config['img_channels']
        ).to(self.device)

        self.g_ema.eval()

        if self.training_config['transfer_learning_model']['enabled']:
            from gan_control.evaluation.inference_class import Inference
            transfer_learning_model, _, _, _ = Inference.retrieve_model(self.training_config['transfer_learning_model']['model_path'])
            self.generator.load_transfer_learning_model(transfer_learning_model)

        accumulate(self.g_ema, self.generator, 0)

        g_reg_ratio = self.training_config['g_reg_every'] / (self.training_config['g_reg_every'] + 1)
        d_reg_ratio = self.training_config['d_reg_every'] / (self.training_config['d_reg_every'] + 1)

        self.g_optim = optim.Adam(
            self.generator.parameters(),
            lr=self.training_config['lr_g'] * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
        self.d_optim = optim.Adam(
            self.discriminator.parameters(),
            lr=self.training_config['lr_d'] * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

        if self.ckpt_config['enabled']:
            _log.info('load model:', self.ckpt_config['ckpt'])
            #print('load model:', self.ckpt_config['ckpt'])

            ckpt = torch.load(self.ckpt_config['ckpt'])

            try:
                ckpt_name = os.path.basename(self.ckpt_config['ckpt'])
                self.training_config['start_iter'] = int(os.path.splitext(ckpt_name)[0])

            except ValueError:
                pass

            self.generator.load_state_dict(ckpt['g'])
            self.discriminator.load_state_dict(ckpt['d'])
            self.g_ema.load_state_dict(ckpt['g_ema'])

            self.g_optim.load_state_dict(ckpt['g_optim'])
            self.d_optim.load_state_dict(ckpt['d_optim'])

        if self.training_config['parallel']:
            self.generator = nn.DataParallel(self.generator)
            self.discriminator = nn.DataParallel(self.discriminator)
            self.g_module = self.generator.module
            self.d_module = self.discriminator.module
        else:
            self.g_module = self.generator
            self.d_module = self.discriminator
        self.g_ema_module = self.g_ema

    def init_data_set(self):
        if self.data_config['data_set_name'] == 'ffhq':
            from gan_control.datasets.ffhq_dataset import get_ffhq_data_loader
            self.loader = get_ffhq_data_loader(self.data_config, batch_size=self.training_config['batch'], training=True, size=self.model_config['size'])
        elif self.data_config['data_set_name'] == 'afhq':
            from gan_control.datasets.afhq_dataset import get_afhq_data_loader
            self.loader = get_afhq_data_loader(self.data_config, batch_size=self.training_config['batch'], training=True, size=self.model_config['size'])
        elif self.data_config['data_set_name'] == 'met-faces':
            from gan_control.datasets.metfaces_dataset import get_metfaces_data_loader
            self.loader = get_metfaces_data_loader(self.data_config, batch_size=self.training_config['batch'], training=True, size=self.model_config['size'])
        else:
            raise ValueError('data_config[data_set_name] = %s (not valid)' % self.data_config['data_set_name'])

    def init_losses(self):
        self.id_embedding_class = None
        self.pose_expression_class = None
        self.pose_orientation_class = None
        self.age_class = None
        self.hair_loss_class = None
        if self.training_config['embedding_loss']['enabled']:
            self.id_embedding_class = LossModelClass(self.training_config['embedding_loss'], loss_name='embedding_loss', mini_batch_size=self.training_config['mini_batch'])
        if 'classification_loss' in self.training_config.keys() and self.training_config['classification_loss']['enabled']:
            self.id_classification_class = LossModelClass(self.training_config['classification_loss'], loss_name='classification_loss', mini_batch_size=self.training_config['mini_batch'])
        if 'dog_id_loss' in self.training_config.keys() and self.training_config['dog_id_loss']['enabled']:
            self.dog_id_class = LossModelClass(self.training_config['dog_id_loss'], loss_name='dog_id_loss', mini_batch_size=self.training_config['mini_batch'])
        if self.training_config['expression_loss']['enabled']:
            self.pose_expression_class = LossModelClass(self.training_config['expression_loss'], loss_name='expression_loss', mini_batch_size=self.training_config['mini_batch'])
        if self.training_config['orientation_loss']['enabled']:
            self.pose_orientation_class = LossModelClass(self.training_config['orientation_loss'], loss_name='orientation_loss', mini_batch_size=self.training_config['mini_batch'])
        if self.training_config['age_loss']['enabled']:
            self.age_class = LossModelClass(self.training_config['age_loss'], loss_name='age_loss', mini_batch_size=self.training_config['mini_batch'])
        if self.training_config['hair_loss']['enabled']:
            self.hair_loss_class = LossModelClass(self.training_config['hair_loss'], loss_name='hair_loss', mini_batch_size=self.training_config['mini_batch'])
        if 'style_loss' in self.training_config and self.training_config['style_loss']['enabled']:
            self.style_loss_class = LossModelClass(self.training_config['style_loss'], loss_name='style_loss', mini_batch_size=self.training_config['mini_batch'])
        if self.training_config['recon_3d_loss']['enabled']:
            self.recon_3d_loss_class = LossModelClass(self.training_config['recon_3d_loss'], loss_name='recon_3d_loss', mini_batch_size=self.training_config['mini_batch'])
            if self.training_config['recon_3d_loss']['id_loss']['enabled']:
                self.recon_3d_id_loss_class = LossModelClass(self.training_config['recon_3d_loss']['id_loss'], loss_name='recon_id_loss', mini_batch_size=self.training_config['mini_batch'], no_model=True)
            if self.training_config['recon_3d_loss']['ex_loss']['enabled']:
                self.recon_3d_ex_loss_class = LossModelClass(self.training_config['recon_3d_loss']['ex_loss'], loss_name='recon_ex_loss', mini_batch_size=self.training_config['mini_batch'], no_model=True)
            if self.training_config['recon_3d_loss']['tex_loss']['enabled']:
                self.recon_3d_tex_loss_class = LossModelClass(self.training_config['recon_3d_loss']['tex_loss'], loss_name='recon_tex_loss', mini_batch_size=self.training_config['mini_batch'], no_model=True)
            if self.training_config['recon_3d_loss']['angles_loss']['enabled']:
                self.recon_3d_angles_loss_class = LossModelClass(self.training_config['recon_3d_loss']['angles_loss'], loss_name='recon_angles_loss', mini_batch_size=self.training_config['mini_batch'], no_model=True)
            if self.training_config['recon_3d_loss']['gamma_loss']['enabled']:
                self.recon_3d_gamma_loss_class = LossModelClass(self.training_config['recon_3d_loss']['gamma_loss'], loss_name='recon_gamma_loss', mini_batch_size=self.training_config['mini_batch'], no_model=True)
            if self.training_config['recon_3d_loss']['xy_loss']['enabled']:
                self.recon_3d_xy_loss_class = LossModelClass(self.training_config['recon_3d_loss']['xy_loss'], loss_name='recon_xy_loss', mini_batch_size=self.training_config['mini_batch'], no_model=True)
            if self.training_config['recon_3d_loss']['z_loss']['enabled']:
                self.recon_3d_z_loss_class = LossModelClass(self.training_config['recon_3d_loss']['z_loss'], loss_name='recon_z_loss', mini_batch_size=self.training_config['mini_batch'], no_model=True)
        self.max_pool_stride_2 = torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=2)

    def init_tensorboard(self, init_dirs):
        self.writer = None
        if self.tensorboard_config['enabled'] and init_dirs:
            self.writer = SummaryWriter(log_dir=os.path.join(self.config.results_dir, 'tensorboard', os.path.split(self.config.save_dir)[-1]))
            self.writer.add_text("parameters", f"# Configuration \n```json\n{json.dumps(self.config.__dict__, indent=2)}\n```", global_step=0)

    def init_evaluation(self):
        self.inception = None
        if self.evaluation_config['fid']['enabled']:
            start_time = time.time()
            _log.info('Loading inception model...')
            self.inception = nn.DataParallel(load_patched_inception_v3()).to(self.device)
            _log.info('Inception model loaded (%.3f sec)' % (time.time() - start_time))
            self.inception.eval()
        self.id_embedding_evaluation_class = None
        self.pose_expression_evaluation_class = None
        self.pose_orientation_evaluation_class = None
        self.age_evaluation_class = None
        self.pose_orientation_evaluation_class = None
        self.hair_evaluation_class = None
        if self.training_config['embedding_loss']['enabled']:
            self.id_embedding_evaluation_class = self.id_embedding_class
        elif 'embedding_loss' in self.evaluation_config['separability']['losses']:
            self.id_embedding_evaluation_class = LossModelClass(self.training_config['embedding_loss'], loss_name='embedding_loss', mini_batch_size=self.training_config['mini_batch'])
        if self.training_config['expression_loss']['enabled']:
            self.pose_expression_evaluation_class = self.pose_expression_class
        elif 'expression_loss' in self.evaluation_config['separability']['losses'] or \
                self.evaluation_config['expression_bar']['enabled']:
            self.pose_expression_evaluation_class = LossModelClass(self.training_config['expression_loss'], loss_name='expression_loss', mini_batch_size=self.training_config['mini_batch'])
        if self.training_config['orientation_loss']['enabled']:
            self.pose_orientation_evaluation_class = self.pose_orientation_class
        elif 'orientation_loss' in self.evaluation_config['separability']['losses'] or \
                self.evaluation_config['orientation_hist']['enabled']:
            self.pose_orientation_evaluation_class = LossModelClass(self.training_config['orientation_loss'], loss_name='orientation_loss', mini_batch_size=self.training_config['mini_batch'])
        if self.training_config['age_loss']['enabled']:
            self.age_evaluation_class = self.age_class
        elif 'age_loss' in self.evaluation_config['separability']['losses']:
            self.age_evaluation_class = LossModelClass(self.training_config['age_loss'], loss_name='age_loss', mini_batch_size=self.training_config['mini_batch'])
        if self.training_config['hair_loss']['enabled']:
            self.hair_evaluation_class = self.hair_loss_class
        elif 'hair_loss' in self.evaluation_config['separability']['losses']:
            self.hair_evaluation_class = LossModelClass(self.training_config['hair_loss'], loss_name='hair_loss', mini_batch_size=self.training_config['mini_batch'])

    def dry_run(self):
        _log.info('Dry run...')
        none_g_grads = set()
        test_in = torch.randn(1, self.model_config['latent_size'], device=self.device)
        if self.training_config['parallel_grad_regularize_step'] and False:
            fake, grad = self.generator([test_in], return_grad=True)
            path = self.g_path_regularize_grad(grad, 0)
        else:
            fake, latent = self.g_module([test_in], return_latents=True)
            path = self.g_path_regularize(fake, latent, 0)
        path[0].backward()
        for n, p in self.generator.named_parameters():
            if p.grad is None:
                none_g_grads.add(n)

        test_in = torch.randn(1, self.model_config['img_channels'], self.model_config['size'], self.model_config['size'], requires_grad=True, device=self.device)
        pred, ver = self.d_module(test_in)
        r1_loss = self.d_r1_loss(pred, test_in)
        r1_loss.backward()
        none_d_grads = set()
        for n, p in self.discriminator.named_parameters():
            if p.grad is None:
                none_d_grads.add(n)

        self.none_g_grads = none_g_grads
        self.none_d_grads = none_d_grads
        _log.info('Dry run finished')

    def train(self):
        self.mean_path_length = 0

        self.accum = 0.5 ** (self.training_config['batch'] / self.training_config['g_moving_average'])
        self.ada_augment = torch.tensor([0.0, 0.0]).cuda()
        self.ada_aug_p = self.training_config['augment']['p'] if self.training_config['augment']['p'] > 0 else 0.0
        self.ada_aug_step = self.training_config['augment']['ada_target'] / self.training_config['augment']['ada_length']
        self.r_t_stat = 0
        self.tracker.evaluation_dict['ada_aug_p'] = self.ada_aug_p
        self.tracker.evaluation_dict['r_t_stat'] = self.r_t_stat


        self.pbar = tqdm(range(self.training_config['iter']), initial=self.training_config['start_iter'], dynamic_ncols=True, smoothing=0.01)
        for idx in self.pbar:
            self.tracker.mark_start_iter()

            i = idx + self.training_config['start_iter']
            if i > self.training_config['iter']:
                _log.info('load Done!:')
                #print('Done!')
                break

            self.discriminator_update(i)
            # with torch.autograd.set_detect_anomaly(True):
            self.generator_update(i)

            self.end_iter_update(i)

    def generator_update(self, i):
        requires_grad(self.generator, True)
        requires_grad(self.discriminator, False)
        noise = mixing_noise(self.training_config['batch'], self.model_config['latent_size'], self.training_config['mixing'], self.device)
        mini_noise_inputs = make_mini_batch_from_noise(noise, self.training_config['batch'], self.training_config['mini_batch'])

        self.generator_step(mini_noise_inputs, i)

        g_regularize = i % self.training_config['g_reg_every'] == 0
        if g_regularize:
            self.generator_regularize_step()

        accumulate(self.g_ema_module, self.g_module, self.accum)

    def reset_losses_for_evaluation_dict(self):
        self.tracker.evaluation_dict['g_adv_loss'] = 0

        if self.training_config['embedding_loss']['enabled']:
            self.tracker.evaluation_dict['g_emb_loss'] = 0
        if self.training_config['age_loss']['enabled']:
            self.tracker.evaluation_dict['g_age_loss'] = 0
        if self.training_config['expression_loss']['enabled']:
            self.tracker.evaluation_dict['expression_loss'] = 0
        if self.training_config['orientation_loss']['enabled']:
            self.tracker.evaluation_dict['g_orientation_loss'] = 0
        if self.training_config['hair_loss']['enabled']:
            self.tracker.evaluation_dict['g_hair_loss'] = 0
        if 'dog_id_loss' in self.training_config.keys() and self.training_config['dog_id_loss']['enabled']:
            self.tracker.evaluation_dict['g_dog_id_loss'] = 0
        if 'classification_loss' in self.training_config.keys() and self.training_config['classification_loss']['enabled']:
            self.tracker.evaluation_dict['g_classification_loss'] = 0
        if 'style_loss' in self.training_config.keys() and self.training_config['style_loss']['enabled']:
            self.tracker.evaluation_dict['g_style_loss'] = 0

        if self.training_config['recon_3d_loss']['enabled']:
            if self.training_config['recon_3d_loss']['id_loss']['enabled']:
                self.tracker.evaluation_dict['id_loss'] = 0
            if self.training_config['recon_3d_loss']['ex_loss']['enabled']:
                self.tracker.evaluation_dict['ex_loss'] = 0
            if self.training_config['recon_3d_loss']['tex_loss']['enabled']:
                self.tracker.evaluation_dict['tex_loss'] = 0
            if self.training_config['recon_3d_loss']['angles_loss']['enabled']:
                self.tracker.evaluation_dict['angles_loss'] = 0
            if self.training_config['recon_3d_loss']['gamma_loss']['enabled']:
                self.tracker.evaluation_dict['gamma_loss'] = 0
            if self.training_config['recon_3d_loss']['xy_loss']['enabled']:
                self.tracker.evaluation_dict['xy_loss'] = 0
            if self.training_config['recon_3d_loss']['z_loss']['enabled']:
                self.tracker.evaluation_dict['z_loss'] = 0

    def generator_step(self, mini_noise_inputs, iter_num):
        self.reset_losses_for_evaluation_dict()
        num_of_mini_batches_in_batch = len(mini_noise_inputs)
        self.generator.zero_grad()
        for k, mini_noise in enumerate(mini_noise_inputs):
            if not self.model_config['vanilla']:
                if self.training_config['mini_batch_mode'] == 'random':
                    self.batch_utils.randomize_places_in_batch()
                mini_noise = self.batch_utils.re_arrange_z(mini_noise, k)
            inject_noise = None
            if self.model_config['g_noise_mode'] == 'same_for_same_id' and not self.model_config['vanilla']:
                inject_noise = self.generator.module.make_noise(batch_size=self.training_config['mini_batch'], device=self.device)
                inject_noise = self.batch_utils.re_arrange_inject_noise(inject_noise, k)
            fake_img, _ = self.generator(mini_noise, noise=inject_noise)
            if self.training_config['augment']['enabled']:
                fake_for_discriminator, _ = augment(fake_img, self.ada_aug_p)
            else:
                fake_for_discriminator = fake_img

            fake_pred, fake_ver = self.discriminator(fake_for_discriminator)

            g_loss = self.g_nonsaturating_loss(fake_pred).div_(num_of_mini_batches_in_batch)
            self.tracker.evaluation_dict['g_adv_loss'] += g_loss.item()

            if not self.model_config['vanilla']:
                id_loss = self.calc_id_losses(fake_img, k, num_of_mini_batches_in_batch)
                pose_loss = self.calc_pose_losses(fake_img, k, num_of_mini_batches_in_batch, iter_num)
                g_loss = g_loss + id_loss + pose_loss
            g_loss.backward()
        self.g_optim.step()

    def calc_id_losses(self, fake_img, k, num_of_mini_batches_in_batch):
        id_loss = 0
        if self.training_config['embedding_loss']['enabled']:
            id_embedding_perceptual_features = self.id_embedding_class.calc_features(fake_img)
            same_id_features, not_same_id_features = self.batch_utils.extract_same_not_same_from_list(id_embedding_perceptual_features, self.training_config['embedding_loss']['same_group_name'])
            embedding_perceptual_loss = self.id_embedding_class.calc_mini_batch_loss(last_layer_same_features=same_id_features, last_layer_not_same_features=not_same_id_features)
            embedding_perceptual_loss.div_(num_of_mini_batches_in_batch)
            self.tracker.evaluation_dict['g_emb_loss'] += embedding_perceptual_loss.item()
            id_loss = id_loss + embedding_perceptual_loss
        if self.training_config['age_loss']['enabled']:
            age_perceptual_features = self.age_class.calc_features(fake_img)
            same_age_features, not_same_age_features = self.batch_utils.extract_same_not_same_from_list(age_perceptual_features, self.training_config['age_loss']['same_group_name'])
            age_perceptual_loss = self.age_class.calc_mini_batch_loss(last_layer_same_features=same_age_features, last_layer_not_same_features=not_same_age_features)
            age_perceptual_loss.div_(num_of_mini_batches_in_batch)
            self.tracker.evaluation_dict['g_age_loss'] += age_perceptual_loss.item()
            id_loss = id_loss + age_perceptual_loss
        if 'classification_loss' in self.training_config.keys() and self.training_config['classification_loss']['enabled']:
            perceptual_features = self.id_classification_class.calc_features(fake_img)
            same_features, not_same_features = self.batch_utils.extract_same_not_same_from_list(perceptual_features, self.training_config['classification_loss']['same_group_name'])
            perceptual_loss = self.id_classification_class.calc_mini_batch_loss(last_layer_same_features=same_features, last_layer_not_same_features=not_same_features)
            perceptual_loss.div_(num_of_mini_batches_in_batch)
            self.tracker.evaluation_dict['g_classification_loss'] += perceptual_loss.item()
            id_loss = id_loss + perceptual_loss
        if 'dog_id_loss' in self.training_config.keys() and self.training_config['dog_id_loss']['enabled']:
            perceptual_features = self.dog_id_class.calc_features(fake_img)
            same_features, not_same_features = self.batch_utils.extract_same_not_same_from_list(perceptual_features, self.training_config['dog_id_loss']['same_group_name'])
            perceptual_loss = self.dog_id_class.calc_mini_batch_loss(last_layer_same_features=same_features, last_layer_not_same_features=not_same_features)
            perceptual_loss.div_(num_of_mini_batches_in_batch)
            self.tracker.evaluation_dict['g_dog_id_loss'] += perceptual_loss.item()
            id_loss = id_loss + perceptual_loss

        return id_loss

    def calc_pose_losses(self, fake_img, k, num_of_mini_batches_in_batch, iter_num):
        pose_loss = 0
        if self.training_config['recon_3d_loss']['enabled']:
            pose_features = self.recon_3d_loss_class.calc_features(fake_img)
            id_futures, ex_futures, tex_futures, angles_futures, gamma_futures, xy_futures, z_futures = self.recon_3d_loss_class.skeleton_model.module.extract_futures_from_vec(pose_features)
            if self.training_config['recon_3d_loss']['id_loss']['enabled']:
                same_features, not_same_features = self.batch_utils.extract_same_not_same_from_list(id_futures, self.training_config['recon_3d_loss']['id_loss']['same_group_name'])
                recon_3d_loss = self.recon_3d_id_loss_class.calc_mini_batch_loss(last_layer_same_features=same_features, last_layer_not_same_features=not_same_features)
                recon_3d_loss.div_(num_of_mini_batches_in_batch)
                self.tracker.evaluation_dict['id_loss'] += recon_3d_loss.item()
                pose_loss = pose_loss + recon_3d_loss
            if self.training_config['recon_3d_loss']['ex_loss']['enabled']:
                same_features, not_same_features = self.batch_utils.extract_same_not_same_from_list(ex_futures, self.training_config['recon_3d_loss']['ex_loss']['same_group_name'])
                recon_3d_loss = self.recon_3d_ex_loss_class.calc_mini_batch_loss(last_layer_same_features=same_features, last_layer_not_same_features=not_same_features)
                recon_3d_loss.div_(num_of_mini_batches_in_batch)
                self.tracker.evaluation_dict['ex_loss'] += recon_3d_loss.item()
                pose_loss = pose_loss + recon_3d_loss
            if self.training_config['recon_3d_loss']['tex_loss']['enabled']:
                same_features, not_same_features = self.batch_utils.extract_same_not_same_from_list(tex_futures, self.training_config['recon_3d_loss']['tex_loss']['same_group_name'])
                recon_3d_loss = self.recon_3d_tex_loss_class.calc_mini_batch_loss(last_layer_same_features=same_features, last_layer_not_same_features=not_same_features)
                recon_3d_loss.div_(num_of_mini_batches_in_batch)
                self.tracker.evaluation_dict['tex_loss'] += recon_3d_loss.item()
                pose_loss = pose_loss + recon_3d_loss
            if self.training_config['recon_3d_loss']['angles_loss']['enabled']:
                same_features, not_same_features = self.batch_utils.extract_same_not_same_from_list(angles_futures, self.training_config['recon_3d_loss']['angles_loss']['same_group_name'])
                recon_3d_loss = self.recon_3d_angles_loss_class.calc_mini_batch_loss(last_layer_same_features=same_features, last_layer_not_same_features=not_same_features)
                recon_3d_loss.div_(num_of_mini_batches_in_batch)
                self.tracker.evaluation_dict['angles_loss'] += recon_3d_loss.item()
                pose_loss = pose_loss + recon_3d_loss
            if self.training_config['recon_3d_loss']['gamma_loss']['enabled']:
                same_features, not_same_features = self.batch_utils.extract_same_not_same_from_list(gamma_futures, self.training_config['recon_3d_loss']['gamma_loss']['same_group_name'])
                recon_3d_loss = self.recon_3d_gamma_loss_class.calc_mini_batch_loss(last_layer_same_features=same_features, last_layer_not_same_features=not_same_features)
                recon_3d_loss.div_(num_of_mini_batches_in_batch)
                self.tracker.evaluation_dict['gamma_loss'] += recon_3d_loss.item()
                pose_loss = pose_loss + recon_3d_loss
            if self.training_config['recon_3d_loss']['xy_loss']['enabled']:
                same_features, not_same_features = self.batch_utils.extract_same_not_same_from_list(xy_futures, self.training_config['recon_3d_loss']['xy_loss']['same_group_name'])
                recon_3d_loss = self.recon_3d_xy_loss_class.calc_mini_batch_loss(last_layer_same_features=same_features, last_layer_not_same_features=not_same_features)
                recon_3d_loss.div_(num_of_mini_batches_in_batch)
                self.tracker.evaluation_dict['xy_loss'] += recon_3d_loss.item()
                pose_loss = pose_loss + recon_3d_loss
            if self.training_config['recon_3d_loss']['z_loss']['enabled']:
                same_features, not_same_features = self.batch_utils.extract_same_not_same_from_list(z_futures, self.training_config['recon_3d_loss']['z_loss']['same_group_name'])
                recon_3d_loss = self.recon_3d_z_loss_class.calc_mini_batch_loss(last_layer_same_features=same_features, last_layer_not_same_features=not_same_features)
                recon_3d_loss.div_(num_of_mini_batches_in_batch)
                self.tracker.evaluation_dict['z_loss'] += recon_3d_loss.item()
                pose_loss = pose_loss + recon_3d_loss
        if 'style_loss' in self.training_config.keys() and self.training_config['style_loss']['enabled']:
            perceptual_features = self.style_loss_class.calc_features(fake_img)
            same_features, not_same_features = self.batch_utils.extract_same_not_same_from_list(perceptual_features, self.training_config['style_loss']['same_group_name'])
            perceptual_loss = self.style_loss_class.calc_mini_batch_loss(last_layer_same_features=same_features, last_layer_not_same_features=not_same_features)
            perceptual_loss.div_(num_of_mini_batches_in_batch)
            self.tracker.evaluation_dict['g_style_loss'] += perceptual_loss.item()
            pose_loss = pose_loss + perceptual_loss
        if self.training_config['expression_loss']['enabled']:
            pose_expression_perceptual_features = self.pose_expression_class.calc_features(fake_img)
            same_expression_features, not_same_expression_features = self.batch_utils.extract_same_not_same_from_list(pose_expression_perceptual_features, self.training_config['expression_loss']['same_group_name'])
            expression_perceptual_loss = self.pose_expression_class.calc_mini_batch_loss(last_layer_same_features=same_expression_features, last_layer_not_same_features=not_same_expression_features)
            expression_perceptual_loss.div_(num_of_mini_batches_in_batch)
            self.tracker.evaluation_dict['expression_loss'] += expression_perceptual_loss.item()
            pose_loss = pose_loss + expression_perceptual_loss
        if self.training_config['orientation_loss']['enabled']:
            pose_orientation_perceptual_features = self.pose_orientation_class.calc_features(fake_img)
            same_orientation_features, not_same_orientation_features = self.batch_utils.extract_same_not_same_from_list(pose_orientation_perceptual_features, self.training_config['orientation_loss']['same_group_name'])
            orientation_perceptual_loss = self.pose_orientation_class.calc_mini_batch_loss(last_layer_same_features=same_orientation_features, last_layer_not_same_features=not_same_orientation_features)
            orientation_perceptual_loss.div_(num_of_mini_batches_in_batch)
            self.tracker.evaluation_dict['g_orientation_loss'] += orientation_perceptual_loss.item()
            pose_loss = pose_loss + orientation_perceptual_loss
        if self.training_config['hair_loss']['enabled']:
            pose_hair_perceptual_features = self.hair_loss_class.calc_features(fake_img)
            same_hair_features, not_same_hair_features = self.batch_utils.extract_same_not_same_from_list(pose_hair_perceptual_features, self.training_config['hair_loss']['same_group_name'])
            hair_perceptual_loss = self.hair_loss_class.calc_mini_batch_loss(last_layer_same_features=same_hair_features, last_layer_not_same_features=not_same_hair_features)
            hair_perceptual_loss.div_(num_of_mini_batches_in_batch)
            self.tracker.evaluation_dict['g_hair_loss'] += hair_perceptual_loss.item()
            pose_loss = pose_loss + hair_perceptual_loss

        return pose_loss

    @staticmethod
    def g_freq_loss(x, gaussian_smooth, upper_thres=1, lower_thres=0., same_pose=False):
        smoothed_x = gaussian_smooth(x)
        smooth_im_size_squared = smoothed_x.shape[-1]
        if same_pose:
            dists = torch.norm(smoothed_x[::2] - smoothed_x[1::2], dim=(2, 3), p=2).squeeze(1).div(
                smooth_im_size_squared)
            loss = torch.mean(torch.pow(torch.clamp(dists - lower_thres, min=0.), 2))
        else:
            dists = torch.norm(smoothed_x[::2] - smoothed_x[1::2], dim=(2, 3), p=2).squeeze(1).div(
                smooth_im_size_squared)
            loss = torch.mean(torch.pow(torch.clamp(upper_thres - dists, min=0.), 2))
        return loss

    @staticmethod
    def g_nonsaturating_loss(fake_pred):
        loss = F.softplus(-fake_pred).mean()
        return loss

    def generator_regularize_step(self):
        self.tracker.evaluation_dict['g_path_loss'] = 0
        self.tracker.evaluation_dict['g_path_length'] = 0
        self.tracker.evaluation_dict['g_mean_path_length'] = 0
        self.generator.zero_grad()
        path_batch_size = max(1, self.training_config['batch'] // self.training_config['path_batch_shrink'])
        noise = mixing_noise(path_batch_size, self.model_config['latent_size'], self.training_config['mixing'], self.device)
        mini_noise_inputs = make_mini_batch_from_noise(noise, self.training_config['batch'], self.training_config['mini_batch'])
        for k, mini_noise in enumerate(mini_noise_inputs):
            if self.training_config['parallel_grad_regularize_step']:
                fake_img, grad = self.generator(mini_noise, return_grad=True)
                path_loss, self.mean_path_length, path_lengths = self.g_path_regularize_grad(grad, self.mean_path_length)
            else:
                if self.training_config['parallel']:
                    fake_img, latents = self.g_module(mini_noise, return_latents=True)
                else:
                    fake_img, latents = self.generator(mini_noise, return_latents=True)
                path_loss, self.mean_path_length, path_lengths = self.g_path_regularize(fake_img, latents, self.mean_path_length)
            path_loss.div_(len(mini_noise_inputs))
            weighted_path_loss = self.training_config['path_regularize'] * self.training_config['g_reg_every'] * path_loss

            if self.training_config['path_batch_shrink']:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            set_grad_none(self.g_module, self.none_g_grads)
            self.tracker.evaluation_dict['g_path_loss'] += path_loss.item()
            self.tracker.evaluation_dict['g_path_length'] += path_lengths.mean().item() / len(mini_noise_inputs)
            self.tracker.evaluation_dict['g_mean_path_length'] += self.mean_path_length.item() / len(mini_noise_inputs)

        self.g_optim.step()

    @staticmethod
    def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01, id_w_grad=False, dim_1_shape=1):
        noise = torch.randn_like(fake_img) / math.sqrt(
            fake_img.shape[2] * fake_img.shape[3] * dim_1_shape
        )
        grad, = autograd.grad(
            outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
        )
        if id_w_grad:
            _, grad = torch.chunk(grad, 2, dim=2)
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
        path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
        path_penalty = (path_lengths - path_mean).pow(2).mean()
        return path_penalty, path_mean.detach(), path_lengths


    @staticmethod
    def g_path_regularize_grad(grad, mean_path_length, decay=0.01, id_w_grad=False):
        if id_w_grad:
            _, grad = torch.chunk(grad, 2, dim=2)
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
        path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
        path_penalty = (path_lengths - path_mean).pow(2).mean()
        return path_penalty, path_mean.detach(), path_lengths

    def discriminator_update(self, i):
        real_img, _ = next(self.loader)
        real_img = real_img.to(self.device)
        mini_real_inputs = real_img.chunk(self.training_config['batch'] // self.training_config['mini_batch'])

        requires_grad(self.generator, False)
        requires_grad(self.discriminator, True)
        noise = mixing_noise(self.training_config['batch'], self.model_config['latent_size'], self.training_config['mixing'], self.device)
        # print('discriminator_update noise: %s' % str(noise))
        mini_noise_inputs = make_mini_batch_from_noise(noise, self.training_config['batch'], self.training_config['mini_batch'])

        d_step = i % self.training_config['d_every'] == 0
        if d_step:
            self.discriminator_step(mini_noise_inputs, mini_real_inputs)

        d_regularize = i % self.training_config['d_reg_every'] == 0
        if d_regularize:
            self.discriminator_regularize_step(mini_real_inputs)

    def discriminator_step(self, mini_noise_inputs, mini_real_inputs):
        self.tracker.evaluation_dict['d_loss'] = 0
        self.discriminator.zero_grad()
        real_ver_list = []
        for k, (mini_real_img, mini_noise) in enumerate(zip(mini_real_inputs, mini_noise_inputs)):
            fake_img, _ = self.generator(mini_noise)
            if self.training_config['augment']['enabled']:
                mini_real_img, _ = augment(mini_real_img, self.ada_aug_p)
                fake_img, _ = augment(fake_img, self.ada_aug_p)

            fake_pred, fake_ver = self.discriminator(fake_img)
            real_pred, real_ver = self.discriminator(mini_real_img)
            d_loss = self.d_logistic_loss(real_pred, fake_pred)
            d_loss.div_(len(mini_real_img))
            self.tracker.evaluation_dict['d_loss'] += d_loss.item()
            real_ver_list.append(real_ver)

            d_loss.backward(retain_graph=True)

            #loss_dict['real_score'] += real_pred.mean().item() / len(mini_real_inputs)
            #loss_dict['fake_score'] += fake_pred.mean().item() / len(mini_real_inputs)

        self.d_optim.step()

        self.ada_augment += torch.tensor(
            (torch.sign(real_pred).sum().item(), real_pred.shape[0]), device=self.device
        )

        if self.ada_augment[1] > 255:
            pred_signs, n_pred = self.ada_augment.tolist()

            self.r_t_stat = pred_signs / n_pred

            if self.training_config['augment']['enabled'] and self.training_config['augment']['p'] == 0:
                if self.r_t_stat > self.training_config['augment']['ada_target']:
                    sign = 1
                else:
                    sign = -1

                self.ada_aug_p += sign * self.ada_aug_step * n_pred
                self.ada_aug_p = min(1, max(0, self.ada_aug_p))
                self.tracker.evaluation_dict['ada_aug_p'] = self.ada_aug_p
            self.ada_augment.mul_(0)
        self.tracker.evaluation_dict['r_t_stat'] = self.r_t_stat

    @staticmethod
    def d_logistic_loss(real_pred, fake_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def discriminator_regularize_step(self, mini_real_inputs):
        self.tracker.evaluation_dict['d_r1_loss'] = 0
        self.discriminator.zero_grad()
        for k, mini_real_img in enumerate(mini_real_inputs):
            mini_real_img.requires_grad = True
            real_pred, real_ver = self.discriminator(mini_real_img)
            r1_loss = self.d_r1_loss(real_pred, mini_real_img)
            r1_loss.div_(len(mini_real_inputs))
            self.tracker.evaluation_dict['d_r1_loss'] += r1_loss.item()
            (self.training_config['r1'] / 2 * r1_loss * self.training_config['d_reg_every'] + 0 * real_pred[0]).backward()

            set_grad_none(self.discriminator, self.none_d_grads)
            self.tracker.evaluation_dict['d_r1_loss'] += r1_loss.item()

        self.d_optim.step()

    def d_r1_loss(self, real_pred, real_img):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    def end_iter_update(self, i):
        self.set_pbar(i)

        if i % self.training_config['min_evaluate_interval'] == 0 or (self.training_config['debug'] and i % 10 == 0):
            self.evaluate(i)
        if i % self.training_config['save_images_interval'] == 0 or (self.training_config['debug'] and i % 100 == 0):
            self.save_images(i)
        if i % self.training_config['save_nets_interval'] == 0 and not self.training_config['debug']:
            self.save_nets(i)
            if self.evaluation_config['fid']['enabled'] and self.tracker.is_best_fid():
                self.save_nets(i, best_fid=True)
        if (i % 200 == 0 or (self.training_config['debug'] and i % 10 == 0)) and self.monitor_config['enabled']:
            self.csv_monitor.update(i)

    def evaluate(self, i):
        self.tracker.evaluate(
            i,
            self.g_ema,
            graph_save_path=os.path.join(self.config.save_dir, 'graphs'),
            buckets_save_path=os.path.join(self.config.save_dir, 'buckets'),
            debug=self.training_config["debug"],
            id_embedding_class=self.id_embedding_evaluation_class,
            pose_orientation_class=self.pose_orientation_evaluation_class,
            pose_expression_class=self.pose_expression_evaluation_class,
            batch_utils=self.batch_utils,
            training_config=self.training_config
        )
        self.tracker.write_stats(i)

    def set_pbar(self, i):
        self.pbar.set_description((
                'd: %.4f; g: %.4f; r1:%.4f; path:%.4f mean path:%.4f; ada:%.4f, ada_r_t:%.4f' % (
                self.tracker.evaluation_dict['d_loss'],
                self.tracker.evaluation_dict['g_adv_loss'],
                self.tracker.evaluation_dict['d_r1_loss'],
                self.tracker.evaluation_dict['g_path_loss'],
                self.tracker.evaluation_dict['g_path_length'],
                self.tracker.evaluation_dict['ada_aug_p'],
                self.tracker.evaluation_dict['r_t_stat']
            )
        ))

    def save_matrix(self, name, matrix_path, same_chunk_group, i, same_noise_for_all=False):
        image = self.tracker.make_matrix(
            self.g_ema,
            downsample=self.model_config['size'] // 256,
            same_chunk=self.batch_utils.place_in_latent_dict[same_chunk_group],
            same_noise_for_all=same_noise_for_all
        )
        os.makedirs(matrix_path, exist_ok=True)
        image.save(f'{matrix_path}/{str(i).zfill(6)}.jpg')
        _log.info('Saved %s to: %s' % (f'{matrix_path}/{str(i).zfill(6)}.jpg', name))

    def save_images(self, i):
        samples_path = os.path.join(self.config.save_dir, 'images', 'sample')
        self.g_ema.eval()
        image = self.tracker.make_samples(self.g_ema)
        image.save(f'{samples_path}/{str(i).zfill(6)}.png')
        _log.info('Saved sample to: %s' % f'{samples_path}/{str(i).zfill(6)}.png')
        matrix_path = os.path.join(self.config.save_dir, 'images', 'matrix')
        self.save_matrix('matrix', matrix_path, self.training_config['embedding_loss']['same_group_name'], i)
        matrix_path = os.path.join(self.config.save_dir, 'images', 'matrix_same_noise')
        self.save_matrix('matrix same noise', matrix_path, self.training_config['embedding_loss']['same_group_name'], i, same_noise_for_all=True)

        if 'other' in self.batch_utils.sub_group_names:
            matrix_path = os.path.join(self.config.save_dir, 'images', 'other_matrix')
            self.save_matrix('other', matrix_path, 'other', i)
            matrix_path = os.path.join(self.config.save_dir, 'images', 'other_matrix_same_noise')
            self.save_matrix('other same noise', matrix_path, 'other', i, same_noise_for_all=True)

        if self.pose_orientation_evaluation_class is not None:
            orientation_matrix_path = os.path.join(self.config.save_dir, 'images', 'orientation_matrix')
            image = self.tracker.make_orientation_matrix(
                self.g_ema,
                self.pose_orientation_evaluation_class,
                downsample=self.model_config['size'] // 256,
                same_chunk=self.batch_utils.place_in_latent_dict[self.training_config['orientation_loss']['same_group_name']]
            )
            image.save(f'{orientation_matrix_path}/{str(i).zfill(6)}.jpg')
            _log.info('Saved orientation matrix to: %s' % f'{orientation_matrix_path}/{str(i).zfill(6)}.jpg')

        if self.pose_expression_evaluation_class is not None:
            expression_matrix_path = os.path.join(self.config.save_dir, 'images', 'expression_matrix')
            image = self.tracker.make_expression_matrix(
                self.g_ema,
                self.pose_expression_evaluation_class,
                downsample=self.model_config['size'] // 256,
                same_chunk=self.batch_utils.place_in_latent_dict[self.training_config['expression_loss']['same_group_name']]
            )
            image.save(f'{expression_matrix_path}/{str(i).zfill(6)}.jpg')
            _log.info('Saved expression matrix to: %s' % f'{expression_matrix_path}/{str(i).zfill(6)}.jpg')

        if self.age_evaluation_class is not None:
            age_matrix_path = os.path.join(self.config.save_dir, 'images', 'age_matrix')
            image = self.tracker.make_age_matrix(
                self.g_ema,
                self.age_evaluation_class,
                downsample=self.model_config['size'] // 256,
                same_chunk=self.batch_utils.place_in_latent_dict[self.training_config['age_loss']['same_group_name']]
            )
            image.save(f'{age_matrix_path}/{str(i).zfill(6)}.jpg')
            _log.info('Saved age matrix to: %s' % f'{age_matrix_path}/{str(i).zfill(6)}.jpg')

        if self.hair_evaluation_class is not None:
            hair_matrix_path = os.path.join(self.config.save_dir, 'images', 'hair_matrix')
            image = self.tracker.make_matrix(
                self.g_ema,
                downsample=self.model_config['size'] // 256,
                same_chunk=self.batch_utils.place_in_latent_dict[self.training_config['hair_loss']['same_group_name']]
            )
            image.save(f'{hair_matrix_path}/{str(i).zfill(6)}.jpg')
            _log.info('Saved hair matrix to: %s' % f'{hair_matrix_path}/{str(i).zfill(6)}.jpg')

        if self.pose_orientation_evaluation_class is not None and self.pose_expression_evaluation_class is not None:
            attribute_matrix_path = os.path.join(self.config.save_dir, 'images', 'attribute_matrix')
            image = self.tracker.make_attribute_matrix(
                self.g_ema,
                self.pose_orientation_evaluation_class,
                self.pose_expression_evaluation_class,
                downsample=self.model_config['size'] // 256,
                same_chunk=self.batch_utils.place_in_latent_dict[self.training_config['embedding_loss']['same_group_name']]
            )
            image.save(f'{attribute_matrix_path}/{str(i).zfill(6)}.jpg')
            _log.info('Saved attribute matrix to: %s' % f'{attribute_matrix_path}/{str(i).zfill(6)}.jpg')

        for group_name in self.batch_utils.get_ordered_group_names():
            matrix_path = os.path.join(self.config.save_dir, 'images', 'default_%s_matrix' % group_name)
            self.save_matrix('default %s' % group_name, matrix_path, group_name, i)
            matrix_path = os.path.join(self.config.save_dir, 'images', 'default_%s_matrix_same_noise' % group_name)
            self.save_matrix('default %s same noise', matrix_path, group_name, i, same_noise_for_all=True)

    def save_nets(self, i, best_fid=False):
        save_path = os.path.join(self.config.save_dir, 'checkpoint')
        save_path = f'{save_path}//best_fid.pt' if best_fid else f'{save_path}//{str(i).zfill(6)}.pt'
        torch.save(
            {
                'g': self.g_module.state_dict(),
                'd': self.d_module.state_dict(),
                'g_ema': self.g_ema.state_dict(),
                'g_optim': self.g_optim.state_dict(),
                'd_optim': self.d_optim.state_dict(),
            },
            save_path,
        )
        _log.info('Saved model to: %s' % save_path)

    def add_weight_to_name(self, save_name):
        def make_wight_name(cfg_dict, key, new_name):
            print(key)
            weight = cfg_dict[key]['last_layer_weight']
            if weight < 1:
                weight = '0%s' % str(weight).split('.')[-1]
            elif int(weight) == weight:
                weight = '%d' % int(weight)
            else:
                weight = '%s%s' % (str(weight).split('.')[0], str(weight).split('.')[1])
            new_name = '%s%s%s' % (new_name, cfg_dict[key]['same_group_name'][:3], weight)
            return new_name

        new_name = ''
        keys = list(self.config.training_config.keys())
        keys.sort()
        for key in keys:
            if key.split('_')[-1] == 'loss':
                if isinstance(self.config.training_config[key], dict) and \
                        ('enabled' in self.config.training_config[key]) and \
                        self.config.training_config[key]['enabled']:
                    if key == 'recon_3d_loss':
                        recon_3d_loss_dict = self.config.training_config[key]
                        keys3d = list(recon_3d_loss_dict.keys())
                        keys3d.sort()
                        for key3d in keys3d:
                            if key3d.split('_')[-1] == 'loss':
                                if isinstance(recon_3d_loss_dict[key3d], dict) and \
                                        ('enabled' in recon_3d_loss_dict[key3d]) and \
                                        recon_3d_loss_dict[key3d]['enabled']:
                                    new_name = make_wight_name(recon_3d_loss_dict, key3d, new_name)
                    else:
                        new_name = make_wight_name(self.config.training_config, key, new_name)
        if len(save_name) > 0:
            new_name = new_name + '_' + save_name
        _log.info('Constructed name for experiment: %s' % new_name)
        return new_name
