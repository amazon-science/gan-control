# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.nn as nn
import copy
import time
import json
import numpy as np
from torchvision import transforms, utils
from PIL import Image

from gan_control.evaluation.generation import gen_grid, gen_matrix, IterableModel
from gan_control.fid_utils.evaluate_fid import evaluate_fid
from gan_control.evaluation.separability import calc_separability
from gan_control.utils.ploting_utils import plot_hist, plot_graph
from gan_control.utils.pandas_utils import get_kmin
from gan_control.utils.pil_images_utils import create_image_grid_from_image_list, write_text_to_image, get_concat_h
from gan_control.evaluation.orientation import make_orientation_hist, make_orientation_grid, calc_orientation_from_tensor_images, write_orientation_to_image
from gan_control.evaluation.expression import make_expression_bar, make_expression_grid, calc_expression_from_tensor_images, write_expression_to_image
from gan_control.evaluation.age import make_age_hist, make_ages_grid

from gan_control.utils.logging_utils import get_logger

_log = get_logger(__name__)


class Tracker():
    def __init__(
            self,
            latent_samples,
            injection_noise_samples,
            writer,
            inception,
            g_noise_mode,
            fid_config=None,
            separability_config=None,
            orientation_hist_config=None,
            expression_bar_config=None
    ):
        self.latent_samples = latent_samples
        self.injection_noise_samples = injection_noise_samples
        self.writer = writer
        self.inception = inception
        self.fid_config = fid_config
        self.separability_config = separability_config
        self.orientation_hist_config = orientation_hist_config
        self.expression_bar_config = expression_bar_config
        self.same_noise_per_id = True if g_noise_mode == 'same_for_same_id' else False

        self.fids = []
        self.evaluation_dict = {}
        self.to_pil = transforms.ToPILImage()
        self.min_iter_time = 1e6
        self.max_iter_time = 0
        self.start_iter_time = None
        self.iter_time_list = []
        self.time_dict = {}

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


    def make_samples(self, model, use_sample_noise=True):
        if use_sample_noise:
            injection_noise_samples = [self.injection_noise_samples[i].detach().cuda() for i in range(len(self.injection_noise_samples))]
        else:
            injection_noise_samples = None
        return gen_grid(model, self.latent_samples.detach().cuda(), injection_noise=injection_noise_samples, nrow=4)

    def make_matrix(self, model, downsample=None, same_chunk=(256, 512), same_noise_for_all=False):
        return gen_matrix(model, same_noise_per_id=self.same_noise_per_id, downsample=downsample, same_chunk=same_chunk, same_noise_for_all=same_noise_for_all)

    def make_orientation_matrix(self, model, orientation_loss_model, downsample=None, same_chunk=(256, 512)):
        image_tensors = gen_matrix(model, same_noise_per_id=self.same_noise_per_id, return_list=True, same_chunk=same_chunk)
        return make_orientation_grid(orientation_loss_model, image_tensors, nrow=6, downsample=downsample)

    def make_expression_matrix(self, model, expression_loss_model, downsample=None, same_chunk=(256, 512)):
        image_tensors = gen_matrix(model, same_noise_per_id=self.same_noise_per_id, return_list=True, same_chunk=same_chunk)
        return make_expression_grid(expression_loss_model, image_tensors, nrow=6, downsample=downsample)

    def make_age_matrix(self, model, age_loss_model, downsample=None, same_chunk=(256, 512)):
        image_tensors = gen_matrix(model, same_noise_per_id=self.same_noise_per_id, return_list=True, same_chunk=same_chunk)
        return make_ages_grid(age_loss_model, image_tensors, nrow=6, downsample=downsample)

    def make_attribute_matrix(self, model, orientation_loss_model, expression_loss_model, downsample=None, same_chunk=(256, 512)):
        image_tensors = gen_matrix(model, same_noise_per_id=self.same_noise_per_id, return_list=True, same_chunk=same_chunk)
        yaw_predicted, pitch_predicted, roll_predicted = calc_orientation_from_tensor_images(orientation_loss_model, image_tensors)
        expressions = calc_expression_from_tensor_images(expression_loss_model, image_tensors)
        image_tensors = image_tensors.mul(0.5).add(0.5).clamp(min=0., max=1.)
        images = [transforms.ToPILImage()(image_tensors[i]) for i in range(image_tensors.shape[0])]
        images = write_orientation_to_image(images, yaw_predicted, pitch_predicted, roll_predicted)
        images = write_expression_to_image(images, expressions)
        image_grid = create_image_grid_from_image_list(images)
        if downsample is not None:
            width, height = image_grid.size
            image_grid = transforms.Resize((width // downsample, height // downsample), interpolation=Image.BILINEAR)(image_grid)
        return create_image_grid_from_image_list(images)

    def evaluate(
            self,
            iter,
            model,
            graph_save_path,
            buckets_save_path,
            debug=False,
            id_embedding_class=None,
            pose_orientation_class=None,
            pose_expression_class=None,
            age_loss_class=None,
            batch_utils=None,
            training_config=None
    ):
        if self.fid_config["enabled"] and ((debug and iter % 100 == 0) or (iter % self.fid_config["fid_interval"] == 0)) and iter != 0:
            self.fid_evaluation(model, graph_save_path=graph_save_path, debug=debug)
        if self.separability_config["enabled"] and ((debug and iter % 100 == 0) or (iter % self.separability_config["separability_interval"] == 0)) and iter != 0:
            try:
                same_chunks = {
                    'embedding_loss': batch_utils.place_in_latent_dict[training_config['embedding_loss']['same_group_name']],
                    'orientation_loss': batch_utils.place_in_latent_dict[training_config['orientation_loss']['same_group_name']],
                    'expression_loss': batch_utils.place_in_latent_dict[training_config['expression_loss']['same_group_name']],
                    'age_loss': batch_utils.place_in_latent_dict[training_config['age_loss']['same_group_name']],
                    'hair_loss': batch_utils.place_in_latent_dict[training_config['hair_loss']['same_group_name']]
                }
                self.separability_evaluation(
                    model,
                    iter,
                    id_embedding_class=id_embedding_class,
                    pose_orientation_class=pose_orientation_class,
                    pose_expression_class=pose_expression_class,
                    graph_save_path=graph_save_path,
                    buckets_save_path=buckets_save_path,
                    debug=debug,
                    same_chunks=same_chunks
                )
            except:
                _log.warning('There was somthing wrong with the separability calculation')
        if self.orientation_hist_config["enabled"] and pose_orientation_class is not None and ((debug and iter % 100 == 0) or (iter % self.orientation_hist_config["orientation_hist_interval"] == 0)) and iter != 0:
            self.orientation_evaluation(model, pose_orientation_class, graph_save_path=graph_save_path)
        if self.expression_bar_config["enabled"] and pose_expression_class is not None and ((debug and iter % 100 == 0) or (iter % self.expression_bar_config["expression_bar_interval"] == 0)) and iter != 0:
            self.expression_evaluation(model, pose_expression_class, graph_save_path=graph_save_path)
        #if self.age_hist_config["enabled"] and pose_expression_class is not None and ((debug and iter % 100 == 0) or (iter % self.expression_bar_config["expression_bar_interval"] == 0)) and iter != 0:
        #    self.age_evaluation(model, age_loss_class, graph_save_path=graph_save_path)

    def expression_evaluation(self, model, pose_expression_class, graph_save_path=None):
        _log.info('Calculating expression bar:')
        make_expression_bar(
            pose_expression_class,
            generator=IterableModel(nn.DataParallel(model), same_noise_for_same_id=self.same_noise_per_id, batch_size=20),
            number_os_samples=self.expression_bar_config["num_of_samples"],
            batch_size=20,
            title=None,
            save_path=os.path.join(graph_save_path, 'expression_bar.jpg')
        )

    def orientation_evaluation(self, model, pose_orientation_class, graph_save_path=None):
        _log.info('Calculating orientation Hist:')
        make_orientation_hist(
            pose_orientation_class,
            generator=IterableModel(nn.DataParallel(model), same_noise_for_same_id=self.same_noise_per_id, batch_size=20),
            number_os_samples=self.orientation_hist_config["num_of_samples"],
            batch_size=20,
            title=None,
            save_path=os.path.join(graph_save_path, 'orientation_hist.jpg')
        )

    def separability_evaluation(
            self,
            model,
            global_step,
            id_embedding_class=None,
            pose_orientation_class=None,
            pose_expression_class=None,
            age_class=None,
            hair_loss_class=None,
            graph_save_path=None,
            buckets_save_path=None,
            debug=False,
            same_chunks=None
    ):
        _log.info('Calculating separability:')
        if 'embedding_loss' in self.separability_config["losses"] and id_embedding_class is not None:
            _log.info('Calculating ID separability...')
            self.make_separability_for_tag(
                model,
                id_embedding_class,
                'id separability',
                'id',
                same_chunks['embedding_loss'],
                os.path.join(graph_save_path, 'id_separability'),
                buckets_save_path,
                global_step,
                debug,
                last_layer_separability_only=self.separability_config['last_layer_separability_only']
            )
        if 'orientation_loss' in self.separability_config["losses"] and pose_orientation_class is not None:
            _log.info('Calculating Orientation separability...')
            self.make_separability_for_tag(
                model,
                pose_orientation_class,
                'orientation separability',
                'orientation',
                same_chunks['orientation_loss'],
                os.path.join(graph_save_path, 'orientation_separability'),
                buckets_save_path,
                global_step,
                debug,
                last_layer_separability_only=self.separability_config['last_layer_separability_only']
            )
        if 'expression_loss' in self.separability_config["losses"] and pose_expression_class is not None:
            _log.info('Calculating Expression separability...')
            self.make_separability_for_tag(
                model,
                pose_expression_class,
                'expression separability',
                'expression',
                same_chunks['expression_loss'],
                os.path.join(graph_save_path, 'expression_separability'),
                buckets_save_path,
                global_step,
                debug,
                last_layer_separability_only=self.separability_config['last_layer_separability_only']
            )
        if 'age_loss' in self.separability_config["losses"] and age_class is not None:
            _log.info('Calculating Age separability...')
            self.make_separability_for_tag(
                model,
                age_class,
                'age separability',
                'age',
                same_chunks['age_loss'],
                os.path.join(graph_save_path, 'age_separability'),
                buckets_save_path,
                global_step,
                debug,
                last_layer_separability_only=self.separability_config['last_layer_separability_only']
            )
        if 'hair_loss' in self.separability_config["losses"] and hair_loss_class is not None:
            _log.info('Calculating Age separability...')
            self.make_separability_for_tag(
                model,
                hair_loss_class,
                'hair separability',
                'hair',
                same_chunks['hair_loss'],
                os.path.join(graph_save_path, 'hair_separability'),
                buckets_save_path,
                global_step,
                debug,
                last_layer_separability_only=self.separability_config['last_layer_separability_only']
            )

    def make_separability_for_tag(self, model, loss_class, title, tag, same_chunk, save_path, buckets_save_path,
                                 global_step, debug, last_layer_separability_only=False):
        same_not_same_list, image_list = calc_separability(
            nn.DataParallel(model),
            loss_class,
            same_noise_for_same_id=self.same_noise_per_id,
            num_of_samples=100 if debug else self.separability_config["num_of_samples"],
            title=title,
            save_path=save_path,
            return_images=True,
            same_chunk=same_chunk,
            last_layer_separability_only=last_layer_separability_only
        )
        self.evaluation_dict["%s_same_avg" % tag] = np.array(same_not_same_list[-1]['same']).mean()
        self.evaluation_dict["%s_all_not_same_avg" % tag] = np.array(same_not_same_list[-1]['all_not_same']).mean()
        self.evaluation_dict["%s_not_same_2nd_best_avg" % tag] = np.array(same_not_same_list[-1]['not_same']).mean()
        _log.info('%s: same avg: %.3f, not same avg %.3f, not same 2nd best avg %.3f' % (
            tag,
            self.evaluation_dict["%s_same_avg" % tag],
            self.evaluation_dict["%s_all_not_same_avg" % tag],
            self.evaluation_dict["%s_not_same_2nd_best_avg" % tag]
        ))
        self.add_separability_hist_to_writer(
            same_not_same_list[-1]['same'],
            same_not_same_list[-1]['not_same'],
            same_not_same_list[-1]['all_not_same'],
            '%s_embedding' % tag,
            global_step
        )
        self.save_separability_buckets('%s_embedding' % tag, same_not_same_list[-1]['pids_2nd_best_pairs_df'],
                                       image_list, global_step, buckets_save_path)

    @staticmethod
    def save_separability_buckets(name, pids_2nd_best_pairs_df, image_list, global_step, buckets_save_path):
        kmin_df = get_kmin(pids_2nd_best_pairs_df, column_name='distance', k=5)
        for i in range(len(kmin_df)):
            row = kmin_df.iloc[i]
            message = '%d) Sig: %d, Que: %d, distance:%.4f' % (i, int(row['signature']), int(row['queries']), row['distance'])
            _log.info(message)
            pil_image = write_text_to_image(get_concat_h(image_list[int(row['signature'])], image_list[int(row['queries'])]), message)
            pil_image.save(os.path.join(buckets_save_path, '%08d_%s_%02d.jpg' % (global_step, name, i)))

    def add_separability_hist_to_writer(self, dists_same, dists_not_same_2nd_best, dists_not_same_all, separability_name, i):
        self.writer.add_histogram(tag='separability/%s_same' % separability_name, values=np.array(dists_same), global_step=i)
        self.writer.add_histogram(tag='separability/%s_not_same_2nd_best' % separability_name, values=np.array(dists_not_same_2nd_best), global_step=i)
        self.writer.add_histogram(tag='separability/%s_not_same_all' % separability_name, values=np.array(dists_not_same_all), global_step=i)
        # global histogram
        #self.writer.add_histogram(tag='separability/%s_all' % separability_name, values=np.array(dists_same), global_step=i)
        #self.writer.add_histogram(tag='separability/%s_all' % separability_name, values=np.array(dists_not_same_2nd_best), global_step=i)
        #self.writer.add_histogram(tag='separability/%s_all' % separability_name, values=np.array(dists_not_same_all), global_step=i)

    def fid_evaluation(self, model, graph_save_path=None, debug=False):
        _log.info('Calculating FID:')
        fid = evaluate_fid(nn.DataParallel(model), self.inception, 20,
                           100 if debug else self.fid_config["num_of_samples"], 'cuda',
                           self.fid_config["inception_stat_path"])
        _log.info('FID: %.6f' % fid)
        self.fids.append(fid)
        self.evaluation_dict['fid'] = fid
        iters = np.array(range(1, len(self.fids) + 1)) * self.fid_config["fid_interval"]
        # iters = range(self.fid_config["fid_interval"], iter + self.fid_config["fid_interval"], self.fid_config["fid_interval"])
        print(iters)
        plot_graph(
            [self.fids],
            x_arrays=[iters],
            title='FID',
            xlabel='Iteration',
            ylabel='FID',
            annotate_min_and_last=True,
            save_path=os.path.join(graph_save_path, 'fid.jpg')
        )

    def is_best_fid(self):
        if len(self.fids) == 0:
            return False
        else:
            return np.array(self.fids)[-1] == np.array(self.fids).min()

    def write_stats(self, iter):
        for key in self.evaluation_dict.keys():
            self.writer.add_scalars('%s/%s' % (key, key), {key: self.evaluation_dict[key]}, global_step=iter)
        for key in self.time_dict.keys():
            self.writer.add_scalars('%s/%s' % (key, key), {key: self.time_dict[key]}, global_step=iter)