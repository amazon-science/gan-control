# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import json
import time
import math
import shutil

from tqdm import tqdm
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils
import numpy as np

from igt_res_gan.utils.file_utils import read_json, setup_logging_from_args
from igt_res_gan.evaluation.inference_class import Inference
from igt_res_gan.models.controller_model import FcStack
from igt_res_gan.evaluation.generation import gen_grid, gen_matrix, IterableModel
from igt_res_gan.trainer.utils import accumulate, make_noise, mixing_noise, requires_grad, make_mini_batch_from_noise, set_grad_none
from igt_res_gan.losses.loss_model import LossModelClass
from igt_res_gan.utils.logging_utils import get_logger
from igt_res_gan.utils.ploting_utils import plot_graph

_log = get_logger(__name__)


class ControllerTrainer():
    def __init__(self, config_path):
        _log.info('Init Trainer...')
        self.device = "cuda"
        self.init_config(config_path)
        self.init_dirs()
        self.init_models_and_optim()
        self.init_loss()
        self.init_tensorboard()
        self.init_evaluation()
        self.init_data_set()
        _log.info('Trainer initiated')

    def init_config(self, config_path):
        self.config = read_json(config_path, return_obj=True)
        self.config.save_name = os.path.join(self.config.save_name, self.config.model_config['loss'])
        if self.config.training_config['debug']:
            self.config.save_name = self.config.save_name + '_debug'
            self.config.results_dir = self.config.results_dir + '_debug'
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
        _log.warning('TODO: implement config checks')
        _log.info('Config passed checking')

    def init_dirs(self):
        os.makedirs(os.path.join(self.config.save_dir, 'checkpoint'), exist_ok=True)
        os.makedirs(os.path.join(os.path.split(self.config.save_dir)[0], 'generator'), exist_ok=True)
        os.makedirs(os.path.join(os.path.split(self.config.save_dir)[0], 'generator', 'checkpoint'), exist_ok=True)
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
        os.makedirs(os.path.join(self.config.save_dir, 'images', 'gamma_matrix'), exist_ok=True)

    def save_generator_ckpt_to_save_dir(self, ckpt_iter):
        new_generator_dir = os.path.join(os.path.split(self.config.save_dir)[0], 'generator')
        _log.info('Saving generator checkpoint to %s' % new_generator_dir)
        shutil.copyfile(os.path.join(self.training_config['generator_dir'], 'args.json'), os.path.join(new_generator_dir, 'args.json'))
        if ckpt_iter != 'best_fid':
            shutil.copyfile(os.path.join(self.training_config['generator_dir'], 'checkpoint', f'{str(ckpt_iter).zfill(6)}.pt'), os.path.join(new_generator_dir, 'checkpoint', f'{str(ckpt_iter).zfill(6)}.pt'))
        else:
            shutil.copyfile(os.path.join(self.training_config['generator_dir'], 'checkpoint', f'{str(ckpt_iter)}.pt'), os.path.join(new_generator_dir, 'checkpoint', f'{str(ckpt_iter)}.pt'))

    def init_models_and_optim(self):
        self.generator, self.batch_utils, self.generator_config, ckpt_iter = Inference.retrieve_model(self.training_config['generator_dir'])
        self.save_generator_ckpt_to_save_dir(ckpt_iter)
        self.batch_utils.print()
        self.generator.eval().cuda()
        if self.model_config['loss'] in ['gamma_loss']:
            self.loss_config = self.generator_config.training_config['recon_3d_loss'][self.model_config['loss']]
        else:
            self.loss_config = self.generator_config.training_config[self.model_config['loss']]
        self.working_group = self.loss_config['same_group_name']
        _log.info('setting working group to: %s' % self.working_group)
        group_chunk = self.batch_utils.place_in_latent_dict[self.working_group]
        self.group_latent_size = group_chunk[1] - group_chunk[0]

        _log.info('init controller')
        self.fc_controller = FcStack(self.model_config['lr_mlp'], self.model_config['n_mlp'], self.model_config['in_dim'], self.model_config['mid_dim'], self.group_latent_size).cuda()
        self.fc_controller.print()

        reg_ratio = self.training_config['reg_every'] / (self.training_config['reg_every'] + 1)
        _log.info('init optim: lr:%.5f, beta:%s' % (self.training_config['lr'] * reg_ratio, str((0 ** reg_ratio, 0.99 ** reg_ratio))))
        self.fc_optim = optim.Adam(
            self.fc_controller.parameters(),
            lr=self.training_config['lr'] * reg_ratio,
            betas=(0 ** reg_ratio, 0.99 ** reg_ratio),
        )

        if self.training_config['parallel']:
            _log.info('moving to data parallel')
            self.generator = nn.DataParallel(self.generator)
            self.fc_controller = nn.DataParallel(self.fc_controller)
            self.generator_module = self.generator.module
            self.fc_controller_module = self.fc_controller.module
        else:
            self.generator_module = self.generator
            self.fc_controller_module = self.fc_controller

        # self.scheduler = optim.lr_scheduler.StepLR(self.fc_optim, 50, gamma=0.5)

    def init_data_set(self):
        start_iter_time = time.time()
        if self.training_config['generate_controls'] == 'sampled':
            _log.info('Producing data set:%s' % self.training_config['generate_controls'])
            batch = 40
            num_of_batches = 100 if self.training_config['debug'] else 2000
            data_set_controlles = []
            with torch.no_grad():
                for iter in tqdm(range(num_of_batches)):
                    latent = torch.randn(batch, self.config['latent_size'], device='cuda')
                    fake_img, _ = self.generator(latent)
                    pred = self.loss_class.predict(fake_img)
                    data_set_controlles += torch.chunk(pred.cpu().numpy(), dim=0)
            self.data_set_controlles = np.array(data_set_controlles)
            self.data_set_size = len(self.data_set_controlles)
            _log.info('Data set ready, %d controls (%.3f sec)' % (self.data_set_size, time.time() - start_iter_time))
        if self.training_config['generate_controls'] == 'sampled_df':
            from igt_res_gan.datasets.dataframe_dataset import get_dataframe_data_loader
            attribute_dict = {'age_loss': 'age', 'orientation_loss': 'orientation',
                              'expression_loss': 'expression3d' if self.model_config['in_dim'] == 64 else 'expression_q',
                              'hair_loss': 'hair', 'gamma_loss':  'gamma3d'}
            if self.model_config['in_dim'] not in (64, 8) and self.model_config['loss'] == 'expression_loss':
                raise ValueError('in_dim %d for expression_loss not 8 for q or 64 3d' % self.model_config['in_dim'])
            self.dataframe_data_loader = get_dataframe_data_loader(self.training_config['sampled_df_path'], attribute_dict[self.model_config['loss']], batch_size=self.training_config['batch'])
            self.eval_dataframe_data_loader = get_dataframe_data_loader(self.training_config['sampled_df_path'], attribute_dict[self.model_config['loss']], batch_size=50, train=False)
            _log.info('Dataframe data loader ready, (%.3f sec)' % (time.time() - start_iter_time))

    def init_loss(self):
        if self.working_group in ['gamma']:
            self.loss_config['model_path'] = self.generator_config.training_config['recon_3d_loss']['model_path']
            self.loss_class = LossModelClass(self.loss_config, loss_name='recon_gamma_loss', mini_batch_size=self.training_config['batch'])
        else:
            self.loss_class = LossModelClass(self.loss_config, loss_name=self.model_config['loss'], mini_batch_size=self.training_config['batch'])
        if self.training_config['rec_loss'] == 'l1':
            self.rec_loss = torch.nn.L1Loss()
        else:
            self.rec_loss = torch.nn.MSELoss()

    def init_tensorboard(self):
        self.writer = None
        if self.tensorboard_config['enabled']:
            log_dir = os.path.join(self.config.results_dir, 'tensorboard', os.path.split(self.config.save_dir)[-1])
            self.writer = SummaryWriter(log_dir=log_dir)
            self.writer.add_text("parameters", f"# Configuration \n```json\n{json.dumps(self.config.__dict__, indent=2)}\n```", global_step=0)
            _log.info('initiated tensorboard, log dir:%s' % log_dir)

    def init_evaluation(self):
        self.evaluation_dict = {}
        self.sample_latent = torch.randn(self.evaluation_config['sample_batch'], self.model_config['latent_size'], device='cuda')
        self.min_iter_time = 1e6
        self.max_iter_time = 0
        self.start_iter_time = None
        self.iter_time_list = []
        self.time_dict = {}
        self.eval_loss_list = []

    def train(self):
        i = self.training_config['start_iter']
        done = False
        requires_grad(self.generator, False)
        requires_grad(self.fc_controller, True)
        self.pbar = tqdm(range(1 + (self.training_config['iter'] // self.dataframe_data_loader.__len__())), initial=self.training_config['start_iter'], dynamic_ncols=True, smoothing=0.01)
        for epoch in self.pbar:
            for idx, batch in enumerate(self.dataframe_data_loader):
                self.mark_start_iter()
                self.controller_update(batch)
                self.end_iter_update(i)
                if i > self.training_config['iter']:
                    _log.info('load Done!:')
                    done = True
                    break
                i += 1
            if done:
                break

    def controller_update(self, batch):
        controls = batch[0].cuda().float()
        org_latent = batch[1].cuda()
        latent_rec_loss, latent_adv_loss, attribute_rec_loss = 0., 0., 0.
        self.fc_controller.train()
        self.fc_controller.zero_grad()

        pred_latent = self.fc_controller(controls.detach())

        if 'latent_rec' in self.training_config['losses'] or 'latent_adv' in self.training_config['losses']:
            latent_rec_loss, latent_adv_loss = self.calc_latent_rec_adv_loss(org_latent, pred_latent, controls)
        if 'attribute_rec' in self.training_config['losses']:
            attribute_rec_loss = self.calc_attribute_rec_controller_loss(org_latent, pred_latent, controls)

        loss = latent_rec_loss + latent_adv_loss + attribute_rec_loss * self.training_config['attribute_rec_w']
        self.evaluation_dict['loss'] = loss.cpu().item()

        loss.backward()
        self.fc_optim.step()

    def calc_latent_rec_adv_loss(self, org_latent, pred_latent, controls):
        rec_loss, adv_loss = 0., 0.
        if 'latent_rec' in self.training_config['losses']:
            group_chunk = self.batch_utils.place_in_latent_dict[self.working_group]
            latent_w = org_latent[:, group_chunk[0]:group_chunk[1]]
            rec_loss = self.rec_loss(pred_latent, latent_w)
            self.evaluation_dict['latent_rec_loss'] = rec_loss.cpu().item()
        return rec_loss, adv_loss

    def calc_attribute_rec_controller_loss(self, org_latent, pred_latent, controls):
        latent_for_generator = self.re_arrange_latent(org_latent, pred_latent)

        fake_img, _ = self.generator([latent_for_generator], input_is_latent=True)

        pred = self.loss_class.predict(fake_img)
        loss = self.loss_class.controller_criterion(pred, controls)
        self.evaluation_dict['attribute_loss'] = loss.cpu().item()
        return loss

    def generate_controls(self, batch):
        if self.training_config['generate_controls'] == 'random':
            return torch.rand([batch, 3]).cuda()
        elif self.training_config['generate_controls'] == 'sampled':
            self.data_set_controlles
            torch.tensor(self.data_set_controlles(np.random.randint(0, self.data_set_size, batch)))

    def re_arrange_latent(self, org_latent, group_latent):
        group_chunk = self.batch_utils.place_in_latent_dict[self.working_group]
        latent = org_latent.clone()
        latent[:, group_chunk[0]:group_chunk[1]] = group_latent[:, :]
        return latent

    def end_iter_update(self, i):
        self.set_pbar(i)

        if i % self.training_config['min_evaluate_interval'] == 0 or (self.training_config['debug'] and i % 10 == 0):
            #if self.training_config['controller_type'] == 'latent_w' and self.training_config['generate_controls'] == 'sampled_df':
            #    self.latent_w_evaluate(i)
            #else:
            #    self.evaluate(i)
            self.evaluate(i)
        if i % self.training_config['save_images_interval'] == 0 or (self.training_config['debug'] and i % 100 == 0):
            self.save_images(i)
        if i % self.training_config['save_nets_interval'] == 0 and not self.training_config['debug']:
            self.save_nets(i)
        if (i % 200 == 0 or (self.training_config['debug'] and i % 10 == 0)) and self.monitor_config['enabled']:
            self.csv_monitor.update(i)

    def latent_w_evaluate(self, i):
        start_time = time.time()
        _log.info('Start evaluation')
        self.fc_controller.eval()
        loss_agg = 0
        num_of_batches = 5 if self.training_config['debug'] else 25
        with torch.no_grad():
            for iter, batch in enumerate(tqdm(self.eval_dataframe_data_loader, disable=True)):
                if iter == num_of_batches:
                    break
                controls = batch[0].cuda().float()
                latent_w = batch[1].cuda()
                group_chunk = self.batch_utils.place_in_latent_dict[self.working_group]
                latent_w = latent_w[:, group_chunk[0]:group_chunk[1]]
                pred_latent_w = self.fc_controller(controls.detach())
                loss = self.rec_loss(pred_latent_w, latent_w)
                loss_agg += loss.cpu().item()
            eval_loss = loss_agg / (iter + 1)
            self.evaluation_dict['eval_loss'] = eval_loss
        self.fc_controller.train()
        _log.info('Evaluation (%.2f sec):\neval_loss: %.4f' % (time.time() - start_time, eval_loss))

    def evaluate(self, i):
        start_time = time.time()
        _log.info('Start evaluation')
        self.fc_controller.eval()
        eval_batch = 40
        latent_rec_loss_agg = 0.
        attribute_loss_agg = 0.
        num_of_batches = 5 if self.training_config['debug'] else 25
        with torch.no_grad():
            for idx, batch in enumerate(self.eval_dataframe_data_loader):
                if idx == num_of_batches:
                    break
                controls = batch[0].cuda().float()
                latent_w = batch[1].cuda()
                pred_latent_w = self.fc_controller(controls.detach())
                group_chunk = self.batch_utils.place_in_latent_dict[self.working_group]
                if 'latent_rec' in self.training_config['losses']:
                    latent_rec_loss = self.rec_loss(pred_latent_w, latent_w[:, group_chunk[0]:group_chunk[1]])
                    latent_rec_loss_agg += latent_rec_loss.cpu().item()
                if 'attribute_rec' in self.training_config['losses']:
                    latent_for_generator = self.re_arrange_latent(latent_w, pred_latent_w)
                    fake_img, _ = self.generator([latent_for_generator], input_is_latent=True)
                    pred = self.loss_class.predict(fake_img)
                    attribute_loss = self.loss_class.controller_criterion(pred, controls)
                    attribute_loss_agg += attribute_loss.cpu().item()
            self.evaluation_dict['eval_loss'] = (latent_rec_loss_agg + attribute_loss_agg) / (idx + 1)
            self.evaluation_dict['eval_rec_loss'] = latent_rec_loss_agg / (idx + 1)
            self.evaluation_dict['attribute_loss'] = attribute_loss_agg / (idx + 1)
        self.fc_controller.train()
        _log.info('\nEvaluation (%.2f sec):\neval_loss: %.4f\neval_rec_loss: %.4f\nattribute_loss: %.4f' % (
            time.time() - start_time,
            self.evaluation_dict['eval_loss'],
            self.evaluation_dict['eval_rec_loss'],
            self.evaluation_dict['attribute_loss']
        ))
        self.eval_loss_list.append(self.evaluation_dict['eval_loss'])
        if len(self.eval_loss_list) > 1:
            plot_graph(
                [self.eval_loss_list[1:]],
                x_arrays=[list(range(self.training_config['min_evaluate_interval'], self.training_config['min_evaluate_interval']*len(self.eval_loss_list), self.training_config['min_evaluate_interval']))],
                xlabel='iter', ylabel='eval loss',
                save_path=os.path.join(self.config.save_dir, 'graphs', 'eval_loss.png'),
                annotate_min_and_last=True
            )

    def set_pbar(self, i):
        description = 'loss: %.4f' % self.evaluation_dict['loss']
        if 'attribute_loss' in list(self.evaluation_dict.keys()):
            description += '; attribute_loss: %.4f' % self.evaluation_dict['attribute_loss']
        if 'latent_rec_loss' in list(self.evaluation_dict.keys()):
            description += '; latent_rec_loss: %.4f' % self.evaluation_dict['latent_rec_loss']
        if 'eval_loss' in list(self.evaluation_dict.keys()):
            description += '; eval_loss: %.4f' % self.evaluation_dict['eval_loss']
        self.pbar.set_description(description)

    def save_sample(self, i):
        samples_path = os.path.join(self.config.save_dir, 'images', 'sample')
        im_in_row_num = 4
        rows = self.evaluation_config['sample_batch'] // im_in_row_num
        controls = self.generate_controls(im_in_row_num)
        controls = torch.cat([controls for _ in range(rows)], dim=0)
        group_latent = self.fc_controller(controls.detach())
        latent_for_generator = self.re_arrange_latent(self.sample_latent, group_latent)
        fake_img, _ = self.generator([latent_for_generator])
        image = gen_grid(self.generator, latent_for_generator, nrow=4)
        image.save(f'{samples_path}/{str(i).zfill(6)}.png')
        _log.info('Saved sample to: %s' % f'{samples_path}/{str(i).zfill(6)}.png')

        samples_path = os.path.join(self.config.save_dir, 'images', 'sample', f'{samples_path}/vis_{str(i).zfill(6)}.png')
        self.loss_class.last_layer_criterion.visual(self.loss_class, fake_img, save_path=samples_path, target_pred=controls)
        _log.info('Saved vis sample to: %s' % samples_path)

    def save_dual_images(self, i):
        self.fc_controller.eval()
        samples_path = os.path.join(self.config.save_dir, 'images', 'sample')
        im_in_row_num = 4
        controls = []
        latent_ws = []
        for idx in range(im_in_row_num ** 2):
            item = np.random.randint(0, self.eval_dataframe_data_loader.dataset.__len__(), 1)
            batch = self.eval_dataframe_data_loader.dataset[item[0]]
            controls.append(batch[0].cuda().unsqueeze(0).float())
            latent_ws.append(batch[1].cuda().unsqueeze(0))
        noise = self.generator.module.make_noise(batch_size=im_in_row_num ** 2, device='cuda')
        controls = torch.cat(controls, dim=0)
        latent_ws = torch.cat(latent_ws, dim=0)
        group_latent = self.fc_controller(controls.detach())
        pred_latent_ws = self.re_arrange_latent(latent_ws.detach(), group_latent)
        pred_fake_img, _ = self.generator([pred_latent_ws], input_is_latent=True, noise=noise)
        fake_img, _ = self.generator([latent_ws], input_is_latent=True, noise=noise)
        to_grid_list = []
        for real, pred in zip(fake_img.split(1), pred_fake_img.split(1)):
            to_grid_list.append(real)
            to_grid_list.append(pred)
        to_grid_tensor = torch.cat(to_grid_list, dim=0)
        to_grid_tensor = to_grid_tensor.mul(0.5).add(0.5).clamp(min=0., max=1.).cpu()
        image = transforms.ToPILImage()(utils.make_grid(to_grid_tensor, nrow=im_in_row_num))
        image.save(f'{samples_path}/{str(i).zfill(6)}.png')
        _log.info('Saved sample to: %s' % f'{samples_path}/{str(i).zfill(6)}.png')
        self.fc_controller.train()

    def save_images(self, i):
        self.save_dual_images(i)
        # self.save_sample(i)

    def save_nets(self, i):
        save_path = os.path.join(self.config.save_dir, 'checkpoint')
        torch.save(
            {
                'controller': self.fc_controller_module.state_dict(),
                'controller_optim': self.fc_optim.state_dict(),
            },
            f'{save_path}//{str(i).zfill(6)}.pt',
        )
        _log.info('Saved model to: %s' % f'{save_path}//{str(i).zfill(6)}.pt')

    def mark_start_iter(self):
        new_start_iter_time = time.time()
        if self.start_iter_time is not None:
            iter_time = new_start_iter_time - self.start_iter_time
            self.iter_time_list.append(iter_time)
            if iter_time > self.max_iter_time:
                self.max_iter_time = iter_time
                self.time_dict['max_iter_time'] = self.max_iter_time
            if self.min_iter_time > iter_time:
                self.min_iter_time = iter_time
                self.time_dict['min_iter_time'] = self.min_iter_time
            if len(self.iter_time_list) > 200:
                self.avg_of_200_iters = np.array(self.iter_time_list).mean()
                self.time_dict['avg_of_200_iters'] = self.avg_of_200_iters
                self.time_dict['max_of_200_iters'] = np.array(self.iter_time_list).max()
                self.time_dict['min_of_200_iters'] = np.array(self.iter_time_list).min()
                self.iter_time_list = []
                _log.info(json.dumps(self.time_dict))
        self.start_iter_time = new_start_iter_time




















