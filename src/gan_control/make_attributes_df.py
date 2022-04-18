# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import sys
from pathlib import Path
_PWD = Path(__file__).absolute().parent

sys.path.append(str(_PWD.parent))
sys.path.append(os.path.join(str(_PWD.parent.parent), 'face-alignment'))

from face_alignment import FaceAlignment, LandmarksType

from gan_control.trainers.generator_trainer import GeneratorTrainer
from gan_control.evaluation.inference_class import Inference
from gan_control.evaluation.age import calc_age_from_tensor_images
from gan_control.evaluation.orientation import calc_orientation_from_tensor_images
from gan_control.evaluation.expression import calc_expression_from_tensor_images, get_class
from gan_control.evaluation.hair import calc_hair_color_from_images
from gan_control.evaluation.face_alignment_utils.face_alignment_utils import align_tensor_images, load_lm3d


@torch.no_grad()
def make_attributes_df(model, trainer, attributes_df_save_path, batch_size=40, number_of_samples=10000, align_3d=True):
    attributes_df = pd.DataFrame(columns=['latents', 'latents_w', 'emb', 'age', 'orientation', 'expression_q', 'hair', 'gamma3d', 'expression3d', 'orientation3d'])
    lm3D = load_lm3d()
    fa = FaceAlignment(LandmarksType._3D, flip_input=False)
    if align_3d:
        trainer.recon_3d_loss_class.skeleton_model.module.config['center_crop'] = None
        trainer.id_embedding_class.skeleton_model.module.config['center_crop'] = None
    for batch_num in tqdm(range(number_of_samples // batch_size)):
        out, latent, latent_w = model.gen_batch(batch_size=batch_size, normalize=False)
        age = calc_age_from_tensor_images(trainer.age_class, out)
        orientation = calc_orientation_from_tensor_images(trainer.pose_orientation_class, out)
        expression_q = calc_expression_from_tensor_images(trainer.pose_expression_class, out)
        hair_color = calc_hair_color_from_images(trainer.hair_loss_class, out)
        if align_3d:
            out = align_tensor_images(out, fa=fa, lm3D=lm3D)
            recon_3d_features = trainer.recon_3d_loss_class.calc_features(out)
        else:
            recon_3d_features = trainer.recon_3d_loss_class.calc_features(out)
        id_futures, ex_futures, tex_futures, angles_futures, gamma_futures, xy_futures, z_futures = trainer.recon_3d_loss_class.skeleton_model.module.extract_futures_from_vec(recon_3d_features)
        gamma3d = gamma_futures
        expression3d = ex_futures
        orientation3d = angles_futures
        arcface_emb = trainer.id_embedding_class.calc_features(out)[-1]

        for latent_i, latent_w_i, age_i, yaw, pitch, roll, expression_q_i, hair_i, \
            gamma3d_i, expression3d_i, orientation3d_i, arcface_emb_i in zip(
            latent.cpu().split(1),
            latent_w.cpu().split(1),
            age.cpu().split(1),
            orientation[0].cpu().split(1),
            orientation[1].cpu().split(1),
            orientation[2].cpu().split(1),
            expression_q,
            hair_color.cpu().split(1),
            gamma3d[0].cpu().split(1),
            expression3d[0].cpu().split(1),
            orientation3d[0].cpu().split(1),
            arcface_emb.cpu().split(1)
        ):
            df_entry = {
                'latents': latent_i[0].numpy(),
                'latents_w': latent_w_i[0][0].numpy(),
                'age': age_i[0].item(),
                'orientation': np.array([yaw[0].item(), pitch[0].item(), roll[0].item()]),
                'expression_q': expression_q_i,
                'hair': hair_i[0].numpy(),
                'gamma3d': gamma3d_i[0].numpy(),
                'expression3d': expression3d_i[0].numpy(),
                'orientation3d': orientation3d_i[0].numpy(),
                'arcface_emb': arcface_emb_i[0].numpy()
            }
            attributes_df = attributes_df.append(df_entry, ignore_index=True)

            if len(attributes_df.latents) % 50000 == 0:
                os.makedirs(os.path.split(attributes_df_save_path)[0], exist_ok=True)
                attributes_df.to_pickle(attributes_df_save_path)

    os.makedirs(os.path.split(attributes_df_save_path)[0], exist_ok=True)
    attributes_df.to_pickle(attributes_df_save_path)
    return attributes_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='path to gan model dir')
    parser.add_argument('--trainer_model', type=str, default='same_as_model_dir')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--number_of_samples', type=int, default=100000)
    parser.add_argument('--save_path', type=str, default='path to save dir')
    args = parser.parse_args()

    model = Inference(args.model_dir)
    if args.trainer_model == 'same_as_model_dir':
        args.trainer_model = args.model_dir
    config_path = os.path.join(args.trainer_model, 'args.json')
    trainer = GeneratorTrainer(config_path, init_dirs=False)
    make_attributes_df(model, trainer, args.save_path, batch_size=args.batch_size, number_of_samples=args.number_of_samples)