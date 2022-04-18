# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from gan_control.utils.logging_utils import get_logger

_log = get_logger(__name__)


class LossModelClass():
    def __init__(self, config, loss_name='embedding_loss', mini_batch_size=4, device="cuda", no_model=False, parallel=True):
        # _log.info('setting up loss model: %s\n%s' % (loss_name, json.dumps(config, indent=2)))
        _log.info('setting up loss model: %s' % loss_name)
        self.config = config
        self.loss_name = loss_name
        self.parallel = parallel
        if no_model:
            self.skeleton_model = None
        else:
            self.skeleton_model = self.get_net_skeleton().to(device)
            self.skeleton_model.eval()
        self.last_layer_criterion = self.get_last_layer_criterion()
        self.l1 = torch.nn.L1Loss(reduction='mean')
        self.lower_thres = self.config['lower_thres']
        self.upper_thres = self.config['upper_thres']
        self.last_lower_thres = self.config['last_lower_thres']
        self.last_upper_thres = self.config['last_upper_thres']
        self.weights = self.config['intermediate_layers_weights'] + [self.config['last_layer_weight']]
        self.valid_mask = torch.tensor(np.tril(np.ones([mini_batch_size, mini_batch_size]), k=-1)).bool()
        self.focus_on_list = self.config['focus_on_list']

    def get_net_skeleton(self):
        if self.loss_name == 'embedding_loss':
            from gan_control.losses.arc_face.arc_face_skeleton import ArcFaceSkeleton
            net_skeleton = ArcFaceSkeleton(self.config)
        elif self.loss_name == 'orientation_loss':
            from gan_control.losses.deep_head_pose.hopenet_skeleton import HopenetSkeleton
            net_skeleton = HopenetSkeleton(self.config)
        elif self.loss_name == 'expression_loss':
            from gan_control.losses.facial_features_esr.esr9_skeleton import ESR9Skeleton
            net_skeleton = ESR9Skeleton(self.config)
        elif self.loss_name == 'age_loss':
            from gan_control.losses.deep_expectation_age.deep_age_skeleton import DeepAgeSkeleton
            net_skeleton = DeepAgeSkeleton(self.config)
        elif self.loss_name == 'hair_loss':
            from gan_control.losses.hair_loss.hair_skeleton import HairSkeleton
            net_skeleton = HairSkeleton(self.config)
        elif self.loss_name in ['recon_3d_loss', 'recon_id_loss', 'recon_ex_loss', 'recon_tex_loss', 'recon_angles_loss', 'recon_gamma_loss', 'recon_xy_loss', 'recon_z_loss']:
            from gan_control.losses.face3dmm_recon.face3dmm_skeleton import Face3dmmSkeleton
            net_skeleton = Face3dmmSkeleton(self.config)
        elif self.loss_name == 'classification_loss':
            from gan_control.losses.imagenet.imagenet_skeleton import ImageNetSkeleton
            net_skeleton = ImageNetSkeleton(self.config)
        elif self.loss_name == 'style_loss':
            from gan_control.losses.stayle.style_skeleton import StyleSkeleton
            net_skeleton = StyleSkeleton(self.config)
        elif self.loss_name == 'dog_id_loss':
            from gan_control.losses.dogfacenet.dogfacenet_skeleton import DogFaceNetSkeleton
            net_skeleton = DogFaceNetSkeleton(self.config)
        else:
            raise ValueError('self.loss_name = %s (not valid)' % self.loss_name)
        return nn.DataParallel(net_skeleton).eval().cuda() if self.parallel else net_skeleton.eval().cuda()

    def get_last_layer_criterion(self):
        if self.loss_name == 'embedding_loss':
            from gan_control.losses.arc_face.arc_face_criterion import ArcFaceCriterion
            criterion = ArcFaceCriterion()
        elif self.loss_name == 'orientation_loss':
            from gan_control.losses.deep_head_pose.hopenet_criterion import HopenetCriterion
            criterion = HopenetCriterion()
        elif self.loss_name == 'expression_loss':
            from gan_control.losses.facial_features_esr.esr9_criterion import ESR9Criterion
            criterion = ESR9Criterion()
        elif self.loss_name == 'age_loss':
            from gan_control.losses.deep_expectation_age.deep_age_criterion import DeepAgeCriterion
            criterion = DeepAgeCriterion()
        elif self.loss_name == 'hair_loss':
            from gan_control.losses.hair_loss.hair_criterion import HairCriterion
            criterion = HairCriterion()
        elif self.loss_name == 'style_loss':
            from gan_control.losses.stayle.style_criterion import StyleCriterion
            criterion = StyleCriterion()
        elif self.loss_name == 'dog_id_loss':
            from gan_control.losses.dogfacenet.dogfacenet_criterion import DogFaceCriterion
            criterion = DogFaceCriterion()
        elif self.loss_name in ['recon_3d_loss', 'recon_id_loss', 'recon_ex_loss', 'recon_tex_loss', 'recon_angles_loss', 'recon_gamma_loss', 'recon_xy_loss', 'recon_z_loss']:
            from gan_control.losses.face3dmm_recon.face3dmm_criterion import Face3dmmCriterion
            criterion = Face3dmmCriterion(self.loss_name)
        elif self.loss_name == 'classification_loss':
            from gan_control.losses.imagenet.imagenet_criterion import ImageNetCriterion
            criterion = ImageNetCriterion()
        else:
            raise ValueError('self.loss_name = %s (not valid)' % self.loss_name)
        return criterion

    def controller_criterion(self, pred, target):
        return self.last_layer_criterion.controller_criterion(pred, target)

    def predict(self, generator_output_image, features=None):
        if features is None:
            features = self.calc_features(generator_output_image)[-1]
        return self.last_layer_criterion.predict(features)

    def get_net(self):
        if self.parallel:
            return self.skeleton_model.module.net
        else:
            return self.skeleton_model.net

    def calc_features(self, batch):
        return self.skeleton_model(batch)

    def calc_mini_batch_loss(self, last_layer_same_features=None, last_layer_not_same_features=None):
        # TODO cange id to same and pose to not_same
        if last_layer_same_features is None: raise ValueError('last_layer_same_features is None')
        if last_layer_not_same_features is None: raise ValueError('last_layer_not_same_features is None')
        feature_loss = 0

        row_size = last_layer_same_features[0].shape[0] + last_layer_not_same_features[0].shape[0]
        valid_mask = self.get_valid_mask(row_size)
        same_in_last_layer_mask = self.make_same_last_layer_mask(last_layer_same_features[0].shape[0] // 2, valid_mask)
        not_same_in_last_layer_mask = self.make_not_same_last_layer_mask(last_layer_same_features[0].shape[0] // 2, last_layer_not_same_features[0].shape[0] // 2, valid_mask)

        # perceptual
        for feature_num in range(len(last_layer_same_features) - 1):
            if self.weights[feature_num] == 0:
                continue
            features = torch.cat([last_layer_same_features[feature_num], last_layer_not_same_features[feature_num]], dim=0)
            if 'intermediate_criterion_as_last_layer' in self.config and self.config['intermediate_criterion_as_last_layer']:
                # criterion_as_last
                feature_distances = self.last_layer_criterion(features, features)
            else:
                # calc distance (l1)
                signatures = features.unsqueeze(dim=1)
                queries = features.unsqueeze(dim=0)
                feature_distances = torch.abs(signatures - queries).mean(dim=[2,3,4])
            not_same_last_layer_dists = feature_distances[not_same_in_last_layer_mask]
            same_last_layer_dists = feature_distances[same_in_last_layer_mask]
            if self.focus_on_list[feature_num] == 'same_as_last_layer':
                not_same_dists = feature_distances[(~same_in_last_layer_mask) * valid_mask]
                same_dists = same_last_layer_dists
            elif self.focus_on_list[feature_num] == 'not_same_as_last_layer':
                not_same_dists = feature_distances[(~not_same_in_last_layer_mask) * valid_mask]
                same_dists = not_same_last_layer_dists
            else:
                raise ValueError('focus_on_list[%d] = %s' % (feature_num, self.focus_on_list[feature_num]))
            same_loss = torch.clamp(same_dists - self.lower_thres[feature_num], min=0.).mean()
            not_same_loss = torch.clamp(self.upper_thres[feature_num] - not_same_dists, min=0.).mean()
            feature_loss = feature_loss + self.weights[feature_num] * (same_loss + not_same_loss)

        # last layer
        embeddings = torch.cat([last_layer_same_features[-1], last_layer_not_same_features[-1]], dim=0)
        signatures = embeddings
        queries = embeddings
        embedding_distances = self.last_layer_criterion(signatures, queries)
        not_same_last_layer_dists = embedding_distances[not_same_in_last_layer_mask]
        same_last_layer_dists = embedding_distances[same_in_last_layer_mask]

        if self.focus_on_list[-1] == 'same_as_last_layer':
            not_same_dists = embedding_distances[(~same_in_last_layer_mask) * valid_mask]
            same_dists = same_last_layer_dists
        elif self.focus_on_list[-1] == 'not_same_as_last_layer':
            not_same_dists = embedding_distances[(~not_same_in_last_layer_mask) * valid_mask]
            same_dists = not_same_last_layer_dists
        else:
            raise ValueError('focus_on_list[%d] = %s' % (-1, self.focus_on_list[-1]))

        same_loss = torch.clamp(same_dists - self.last_lower_thres, min=0.).mean()
        not_same_loss = torch.clamp(self.last_upper_thres - not_same_dists, min=0.).mean()

        feature_loss = feature_loss + self.weights[-1] * (same_loss + not_same_loss)

        return feature_loss

    @staticmethod
    def make_same_last_layer_mask(num_same_id_pairs, valid_mask):
        same_id_mask = torch.zeros_like(valid_mask)
        for i in range(num_same_id_pairs):
            same_id_mask[2*i+1, 2*i] = 1
        return same_id_mask.bool() * valid_mask

    @staticmethod
    def make_not_same_last_layer_mask(num_same_id_pairs, num_same_pose_pairs, valid_mask):
        same_pose_mask = torch.zeros_like(valid_mask)
        for i in range(num_same_id_pairs, num_same_id_pairs + num_same_pose_pairs):
            same_pose_mask[2 * i + 1, 2 * i] = 1
        return same_pose_mask.bool() * valid_mask

    def get_valid_mask(self, row_size):
        if len(self.valid_mask) == row_size:
            return self.valid_mask
        else:
            self.valid_mask = torch.tensor(np.tril(np.ones([len(self.valid_mask), len(self.valid_mask)]), k=-1)).bool()
            return self.valid_mask

    def calc_same_not_same_list(self, signatures_list, queries_list, signature_pids, queries_pids, last_layer_separability_only=False):
        all_distances_list = self.calc_distances_list(signatures_list, queries_list, last_layer_separability_only=last_layer_separability_only)
        same_mask = signature_pids[:, None] == queries_pids[None, :]
        signature_pids_v, queries_pids_v = np.meshgrid(signature_pids, queries_pids, sparse=False, indexing='ij')
        same_not_same_list = []
        for layer_num, all_distances in enumerate(all_distances_list):
            pids_2nd_best_pairs_df = pd.DataFrame(columns=('signature', 'queries', 'distance'))
            same = []
            not_same = []
            all_not_same = []
            for qid in tqdm(
                    np.arange(all_distances.shape[1]),
                    desc=f"{self.__class__.__name__}: Calculating 2nd best and filtering sessions"
            ):
                same_row_mask = same_mask[:, qid]
                not_same_row_mask = (~same_mask[:, qid])
                same_row = all_distances[same_row_mask, qid]
                not_same_row = all_distances[not_same_row_mask, qid]
                same.extend(same_row)
                distance_2nd_best = np.min(not_same_row)
                index_2nd_best = np.argmin(not_same_row)
                signature = signature_pids_v[not_same_row_mask, qid][index_2nd_best]
                not_same.append(distance_2nd_best)  # takes only the closest not_same (2nd best)
                all_not_same.extend(not_same_row)
                pids_2nd_best_pairs = {'signature': signature, 'queries': queries_pids[qid], 'distance': distance_2nd_best}
                pids_2nd_best_pairs_df = pids_2nd_best_pairs_df.append(pids_2nd_best_pairs, ignore_index=True)
            _log.info("%s: Resulted in [#same=%s, #not_same=%s] (layer: %d)", self.__class__.__name__, len(same), len(not_same), layer_num)
            same_not_same_list.append({'same': same,
                                       'not_same': not_same,
                                       'all_not_same': all_not_same,
                                       'pids_2nd_best_pairs_df': pids_2nd_best_pairs_df})

        return same_not_same_list

    def calc_distances_list(self, signatures_list, queries_list, last_layer_separability_only=False):
        distances_list = []
        for i, (signatures, queries) in enumerate(zip(signatures_list, queries_list)):
            if i+1 < len(signatures_list):
                if last_layer_separability_only:
                    continue
                if 'intermediate_criterion_as_last_layer' in self.config and self.config['intermediate_criterion_as_last_layer']:
                    distances_list.append(self.calc_distances(signatures, queries, self.last_layer_criterion))
                else:
                    distances_list.append(self.calc_distances(signatures, queries, L1Expend()))
            else:
                distances_list.append(self.calc_distances(signatures, queries, self.last_layer_criterion))
        return distances_list

    def calc_distances(self, signatures, queries, distance_fn, batch_size=20):
        queries_chunks = self._make_chunks(queries, batch_size)
        signatures_chunks = self._make_chunks(signatures, batch_size)
        all_distances = self._iterate_chunk_pairs(queries_chunks, signatures_chunks, distance_fn, batch_size)
        return all_distances

    @staticmethod
    def _make_chunks(tensor, batch_size):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        num_chunks = int(np.ceil(tensor.shape[0] / batch_size))
        chunks = torch.chunk(tensor, chunks=num_chunks, dim=0)
        return chunks

    def _iterate_chunk_pairs(self, queries_chunks, signatures_chunks, distance_fn, batch_size):
        _log.info("%s: Computing batch-wise distances (%s)", self.__class__.__name__, str(type(distance_fn)))
        all_distances = []
        with torch.no_grad():
            for queries_batch in tqdm(queries_chunks,
                                      desc=f"{self.__class__.__name__}: Computing distances",
                                      unit_scale=batch_size,
                                      unit="queries"
                                      ):
                distances = []
                for signatures_batch in signatures_chunks:
                    if torch.cuda.is_available():
                        signatures_batch = signatures_batch.cuda()
                        queries_batch = queries_batch.cuda()
                    distances.append(distance_fn(signatures_batch, queries_batch))

                all_distances.append(torch.cat(distances, dim=0))
        _log.info("%s: Collecting results", self.__class__.__name__)
        all_distances = torch.cat(all_distances, dim=1)
        return all_distances.cpu().numpy()


def add_to_feature_list(loss_model, features_list, real_img):
    features = loss_model.calc_features(real_img)
    for feature_num, feature in enumerate(features):
        if batch_num == 0:
            features_list.append(feature.cpu())
        else:
            features_list[feature_num] = torch.cat([features_list[feature_num], feature.cpu()], dim=0)
    return features_list


def get_same_not_same_list(loss_model, features):
    same_not_same_list = loss_model.calc_same_not_same_list(
        [features[i][::2] for i in range(len(features))],
        [features[i][1::2] for i in range(len(features))],
        np.array(range(0, len(features[0]), 2)),
        np.array(range(0, len(features[0]), 2))
    )
    return same_not_same_list


class L1Expend(torch.nn.Module):
    def __init__(self):
        super(L1Expend, self).__init__()

    def __call__(self, signatures: torch.Tensor, queries: torch.Tensor):
        if len(signatures.shape) == 4:
            signatures = signatures.unsqueeze(dim=1)
            queries = queries.unsqueeze(dim=0)
            loss = torch.abs(signatures - queries).mean(dim=[2, 3, 4])
        elif len(signatures.shape) == 2:
            signatures = signatures.unsqueeze(dim=1)
            queries = queries.unsqueeze(dim=0)
            loss = torch.abs(signatures - queries).mean(dim=[2])
        return loss

if __name__ == '__main__':
    import argparse
    from gan_control.datasets.ffhq_dataset import get_ffhq_data_loader
    from gan_control.utils.file_utils import read_json
    from gan_control.utils.ploting_utils import plot_hist

    parser = argparse.ArgumentParser()
    parser.add_argument('--check_embedding_loss', action='store_true')
    parser.add_argument('--check_orientation_loss', action='store_true')
    parser.add_argument('--check_expression_loss', action='store_true')
    parser.add_argument('--check_age_loss', action='store_true')
    parser.add_argument('--check_hair_loss', action='store_true')
    parser.add_argument('--check_gaze_loss', action='store_true')
    parser.add_argument('--check_background_loss', action='store_true')
    parser.add_argument('--check_recon_3d_ex_loss', action='store_true')
    parser.add_argument('--check_recon_3d_angles_loss', action='store_true')
    parser.add_argument('--check_recon_3d_gamma_loss', action='store_true')
    parser.add_argument('--check_classification_loss', action='store_true')
    parser.add_argument('--check_dog_id_loss', action='store_true')
    parser.add_argument('--check_style_loss', action='store_true')
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=40)
    args = parser.parse_args()
    config = read_json(args.config_path, return_obj=True)
    if config.data_config['data_set_name'] == 'ffhq':
        from gan_control.datasets.ffhq_dataset import get_ffhq_data_loader
        loader = get_ffhq_data_loader(config.data_config, batch_size=args.batch_size, training=False, size=config.model_config['size'])
    elif config.data_config['data_set_name'] == 'afhq':
        from gan_control.datasets.afhq_dataset import get_afhq_data_loader
        loader = get_afhq_data_loader(config.data_config, batch_size=args.batch_size, training=False, size=config.model_config['size'])
    elif config.data_config['data_set_name'] == 'met-faces':
        from gan_control.datasets.metfaces_dataset import get_metfaces_data_loader
        loader = get_metfaces_data_loader(config.data_config, batch_size=args.batch_size, training=False, size=config.model_config['size'])
    else:
        raise ValueError('data_config[data_set_name] = %s (not valid)' % config.data_config['data_set_name'])

    embedding_loss_model = None
    orientation_loss_model = None
    expression_loss_model = None
    age_loss_model = None
    hair_loss_model = None
    recon_3d_ex_loss_model = None
    recon_3d_angles_loss_model = None
    recon_3d_gamma_loss_model = None
    recon_3d_loss_model = None
    dog_id_loss_model = None
    classification_loss_model = None
    style_loss_model = None

    if args.check_embedding_loss:
        embedding_loss_model = LossModelClass(config.training_config['embedding_loss'], loss_name='embedding_loss', mini_batch_size=args.batch_size, device="cuda")
    if args.check_orientation_loss:
        orientation_loss_model = LossModelClass(config.training_config['orientation_loss'], loss_name='orientation_loss', mini_batch_size=args.batch_size, device="cuda")
    if args.check_expression_loss:
        expression_loss_model = LossModelClass(config.training_config['expression_loss'], loss_name='expression_loss', mini_batch_size=args.batch_size, device="cuda")
    if args.check_age_loss:
        age_loss_model = LossModelClass(config.training_config['age_loss'], loss_name='age_loss', mini_batch_size=args.batch_size, device="cuda")
    if args.check_hair_loss:
        hair_loss_model = LossModelClass(config.training_config['hair_loss'], loss_name='hair_loss', mini_batch_size=args.batch_size, device="cuda")
    if args.check_recon_3d_ex_loss:
        recon_3d_ex_loss_model = LossModelClass(config.training_config['recon_3d_loss']['ex_loss'], loss_name='recon_ex_loss', mini_batch_size=args.batch_size, device="cuda", no_model=True)
    if args.check_recon_3d_angles_loss:
        recon_3d_angles_loss_model = LossModelClass(config.training_config['recon_3d_loss']['angles_loss'], loss_name='recon_angles_loss', mini_batch_size=args.batch_size, device="cuda", no_model=True)
    if args.check_recon_3d_gamma_loss:
        recon_3d_gamma_loss_model = LossModelClass(config.training_config['recon_3d_loss']['gamma_loss'], loss_name='recon_gamma_loss', mini_batch_size=args.batch_size, device="cuda", no_model=True)
    if args.check_recon_3d_gamma_loss or args.check_recon_3d_angles_loss or args.check_recon_3d_ex_loss:
        recon_3d_loss_model = LossModelClass(config.training_config['recon_3d_loss'], loss_name='recon_3d_loss', mini_batch_size=args.batch_size, device="cuda")
    if args.check_classification_loss:
        classification_loss_model = LossModelClass(config.training_config['classification_loss'], loss_name='classification_loss', mini_batch_size=args.batch_size, device="cuda")
    if args.check_style_loss:
        style_loss_model = LossModelClass(config.training_config['style_loss'], loss_name='style_loss', mini_batch_size=args.batch_size, device="cuda")
    if args.check_dog_id_loss:
        dog_id_loss_model = LossModelClass(config.training_config['dog_id_loss'], loss_name='dog_id_loss', mini_batch_size=args.batch_size, device="cuda")


    embedding_features = []
    orientation_features = []
    expression_features = []
    age_features = []
    hair_features = []
    gaze_features = []
    recon_ex_features = []
    recon_angles_features = []
    recon_gamma_features = []
    background_features = []
    classification_features = []
    style_features = []
    dog_id_features = []

    with torch.no_grad():
        for batch_num in tqdm(range(0, 1000, args.batch_size)):
            real_img, _ = next(loader)
            if args.check_embedding_loss:
                embedding_features = add_to_feature_list(embedding_loss_model, embedding_features, real_img)
            if args.check_orientation_loss:
                orientation_features = add_to_feature_list(orientation_loss_model, orientation_features, real_img)
            if args.check_expression_loss:
                expression_features = add_to_feature_list(expression_loss_model, expression_features, real_img)
            if args.check_age_loss:
                age_features = add_to_feature_list(age_loss_model, age_features, real_img)
            if args.check_hair_loss:
                hair_features = add_to_feature_list(hair_loss_model, hair_features, real_img)
            if args.check_classification_loss:
                classification_features = add_to_feature_list(classification_loss_model, classification_features, real_img)
            if args.check_style_loss:
                style_features = add_to_feature_list(style_loss_model, style_features, real_img)
            if args.check_dog_id_loss:
                dog_id_features = add_to_feature_list(dog_id_loss_model, dog_id_features, real_img)

            if args.check_recon_3d_gamma_loss or args.check_recon_3d_angles_loss or args.check_recon_3d_ex_loss:
                features = recon_3d_loss_model.calc_features(real_img)
                id_futures, ex_futures, tex_futures, angles_futures, gamma_futures, xy_futures, z_futures = recon_3d_loss_model.skeleton_model.module.extract_futures_from_vec(features)
                if args.check_recon_3d_ex_loss:
                    for feature_num, feature in enumerate(ex_futures):
                        if batch_num == 0:
                            recon_ex_features.append(feature.cpu())
                        else:
                            recon_ex_features[feature_num] = torch.cat([recon_ex_features[feature_num], feature.cpu()], dim=0)
                if args.check_recon_3d_angles_loss:
                    for feature_num, feature in enumerate(angles_futures):
                        if batch_num == 0:
                            recon_angles_features.append(feature.cpu())
                        else:
                            recon_angles_features[feature_num] = torch.cat([recon_angles_features[feature_num], feature.cpu()], dim=0)
                if args.check_recon_3d_gamma_loss:
                    for feature_num, feature in enumerate(gamma_futures):
                        if batch_num == 0:
                            recon_gamma_features.append(feature.cpu())
                        else:
                            recon_gamma_features[feature_num] = torch.cat([recon_gamma_features[feature_num], feature.cpu()], dim=0)

    if args.check_embedding_loss:
        same_not_same_list = get_same_not_same_list(embedding_loss_model, embedding_features)
        for i in range(len(same_not_same_list)):
            arrays = [same_not_same_list[i]['same'], same_not_same_list[i]['not_same'], same_not_same_list[i]['all_not_same']]
            plot_hist(arrays, title='embedding_layer_%d' % i, labels=['same', 'not_same_2nd_best', 'all_not_same'], xlabel='Distance', bins=100, ncol=3, percentiles=(0.2, 0.5, 0.8), min_lim=0, max_lim=1000)
    if args.check_orientation_loss:
        same_not_same_list = get_same_not_same_list(orientation_loss_model, orientation_features)
        for i in range(len(same_not_same_list)):
            arrays = [same_not_same_list[i]['same'], same_not_same_list[i]['not_same'], same_not_same_list[i]['all_not_same']]
            plot_hist(arrays, title='orientation_layer_%d' % i, labels=['same', 'not_same_2nd_best', 'all_not_same'], xlabel='Distance', bins=100, ncol=3, percentiles=(0.2, 0.5, 0.8), min_lim=0, max_lim=1000)
    if args.check_expression_loss:
        same_not_same_list = get_same_not_same_list(expression_loss_model, expression_features)
        for i in range(len(same_not_same_list)):
            arrays = [same_not_same_list[i]['same'], same_not_same_list[i]['not_same'], same_not_same_list[i]['all_not_same']]
            plot_hist(arrays, title='expression_layer_%d' % i, labels=['same', 'not_same_2nd_best', 'all_not_same'], xlabel='Distance', bins=100, ncol=3, percentiles=(0.2, 0.5, 0.8), min_lim=0, max_lim=1000)
    if args.check_age_loss:
        same_not_same_list = get_same_not_same_list(age_loss_model, age_features)
        for i in range(len(same_not_same_list)):
            arrays = [same_not_same_list[i]['same'], same_not_same_list[i]['not_same'], same_not_same_list[i]['all_not_same']]
            plot_hist(arrays, title='age_layer_%d' % i, labels=['same', 'not_same_2nd_best', 'all_not_same'], xlabel='Distance', bins=100, ncol=3, percentiles=(0.2, 0.5, 0.8), min_lim=0, max_lim=1000)
    if args.check_classification_loss:
        same_not_same_list = get_same_not_same_list(classification_loss_model, classification_features)
        for i in range(len(same_not_same_list)):
            arrays = [same_not_same_list[i]['same'], same_not_same_list[i]['not_same'], same_not_same_list[i]['all_not_same']]
            plot_hist(arrays, title='classification_layer_%d' % i, labels=['same', 'not_same_2nd_best', 'all_not_same'], xlabel='Distance', bins=100, ncol=3, percentiles=(0.2, 0.5, 0.8), min_lim=0, max_lim=1000)
    if args.check_style_loss:
        same_not_same_list = get_same_not_same_list(style_loss_model, style_features)
        for i in range(len(same_not_same_list)):
            arrays = [same_not_same_list[i]['same'], same_not_same_list[i]['not_same'], same_not_same_list[i]['all_not_same']]
            plot_hist(arrays, title='style_layer_%d' % i, labels=['same', 'not_same_2nd_best', 'all_not_same'], xlabel='Distance', bins=100, ncol=3, percentiles=(0.2, 0.5, 0.8), min_lim=0, max_lim=1000)
    if args.check_dog_id_loss:
        same_not_same_list = get_same_not_same_list(dog_id_loss_model, dog_id_features)
        for i in range(len(same_not_same_list)):
            arrays = [same_not_same_list[i]['same'], same_not_same_list[i]['not_same'], same_not_same_list[i]['all_not_same']]
            plot_hist(arrays, title='dog_id_layer_%d' % i, labels=['same', 'not_same_2nd_best', 'all_not_same'], xlabel='Distance', bins=100, ncol=3, percentiles=(0.2, 0.5, 0.8), min_lim=0, max_lim=1000)
    if args.check_recon_3d_ex_loss:
        same_not_same_list = get_same_not_same_list(recon_3d_ex_loss_model, recon_ex_features)
        for i in range(len(same_not_same_list)):
            arrays = [same_not_same_list[i]['same'], same_not_same_list[i]['not_same'], same_not_same_list[i]['all_not_same']]
            plot_hist(arrays, title='recon_ex_layer_%d' % i, labels=['same', 'not_same_2nd_best', 'all_not_same'], xlabel='Distance', bins=100, ncol=3, percentiles=(0.2, 0.5, 0.8), min_lim=0, max_lim=1000)
    if args.check_recon_3d_angles_loss:
        same_not_same_list = get_same_not_same_list(recon_3d_angles_loss_model, recon_angles_features)
        for i in range(len(same_not_same_list)):
            arrays = [same_not_same_list[i]['same'], same_not_same_list[i]['not_same'], same_not_same_list[i]['all_not_same']]
            plot_hist(arrays, title='recon_angles_layer_%d' % i, labels=['same', 'not_same_2nd_best', 'all_not_same'], xlabel='Distance', bins=100, ncol=3, percentiles=(0.2, 0.5, 0.8), min_lim=0, max_lim=1000)
    if args.check_recon_3d_gamma_loss:
        same_not_same_list = get_same_not_same_list(recon_3d_gamma_loss_model, recon_gamma_features)
        for i in range(len(same_not_same_list)):
            arrays = [same_not_same_list[i]['same'], same_not_same_list[i]['not_same'], same_not_same_list[i]['all_not_same']]
            plot_hist(arrays, title='recon_gamma_layer_%d' % i, labels=['same', 'not_same_2nd_best', 'all_not_same'], xlabel='Distance', bins=100, ncol=3, percentiles=(0.2, 0.5, 0.8), min_lim=0, max_lim=1000)

    if args.check_hair_loss:
        same_not_same_list = hair_loss_model.calc_same_not_same_list(
            [hair_features[i][::2] for i in range(len(hair_features))],
            [hair_features[i][1::2] for i in range(len(hair_features))],
            np.array(range(0, len(hair_features[0]), 2)),
            np.array(range(0, len(hair_features[0]), 2))
        )
        for i in range(len(same_not_same_list)):
            same_not_same_list[i]['same'] = np.array(same_not_same_list[i]['same'])
            same_not_same_list[i]['not_same'] = np.array(same_not_same_list[i]['not_same'])
            same_not_same_list[i]['all_not_same'] = np.array(same_not_same_list[i]['all_not_same'])
            arrays = [list(same_not_same_list[i]['same'][same_not_same_list[i]['same'] > 0.0001]), list(same_not_same_list[i]['not_same'][same_not_same_list[i]['not_same'] > 0.0001]), list(same_not_same_list[i]['all_not_same'][same_not_same_list[i]['all_not_same'] > 0.0001])]
            plot_hist(arrays, show_kde=False, title='hair_layer_%d' % i, labels=['same', 'not_same_2nd_best', 'all_not_same'], xlabel='Distance', bins=100, ncol=3, percentiles=(0.2, 0.5, 0.8), min_lim=-1, max_lim=1000)


    print('Done!')










