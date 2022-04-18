# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from torchvision import transforms, utils
from PIL import Image

from gan_control.utils.ploting_utils import plot_bar, plot_hist
from gan_control.utils.hopenet_utils import softmax_temperature, draw_axis
from gan_control.utils.logging_utils import get_logger
from gan_control.utils.pil_images_utils import create_image_grid_from_image_list, write_text_to_image

_log = get_logger(__name__)


def calc_hair_color_from_images(hair_loss_class, tensor_images):
    features = hair_loss_class.calc_features(tensor_images)[-1]
    return hair_loss_class.last_layer_criterion.predict(features)

def calc_hair_mask_from_images(hair_loss_class, tensor_images):
    with torch.no_grad():
        features_list = hair_loss_class.calc_features(tensor_images)
    features = features_list[-1]
    return features[:,3:,:,:]


def calc_and_add_hair_to_image(hair_loss_class, tensor_images):
    mask = calc_hair_mask_from_images(hair_loss_class, tensor_images)
    b, c, h, w = mask.shape
    tensor_images = F.interpolate(tensor_images, size=(h, w), mode='bilinear', align_corners=True)
    tensor_images = tensor_images.cpu()
    tensor_images[:,2:3,:,:] = tensor_images[:,2:3,:,:].cpu() + mask.cpu()
    tensor_images = tensor_images.mul(0.5).add(0.5).clamp(min=0., max=1.)
    images = [transforms.ToPILImage()(tensor_images[i]) for i in range(tensor_images.shape[0])]
    return images


def make_hair_seg_grid(hair_loss_class, tensor_images, nrow=6, save_path=None, downsample=None):
    pil_images_with_hair = calc_and_add_hair_to_image(hair_loss_class, tensor_images)
    image_grid = create_image_grid_from_image_list(pil_images_with_hair, nrow=nrow)
    if downsample is not None:
        width, height = image_grid.size
        image_grid = transforms.Resize((width // downsample, height // downsample), interpolation=Image.BILINEAR)(image_grid)
    if save_path is not None:
        image_grid.save(save_path)
    return image_grid


def add_colors_to_images(preds, tensor_images, target_pred=None):
    tensor_images = tensor_images.mul(0.5).add(0.5).clamp(min=0., max=1.)
    tensor_images[:, :, :40, :40] = preds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 40, 40).clone()
    if target_pred is not None:
        tensor_images[:, :, :40, 40:45] = torch.zeros_like(tensor_images)[:, :, :40, 40:45]
        tensor_images[:, :, :40, 45:85] = target_pred.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 40, 40).clone()
    tensor_grid = utils.make_grid(tensor_images, nrow=6)
    return transforms.ToPILImage()(tensor_grid.cpu())


def make_hair_color_grid(hair_loss_class, tensor_images, nrow=6, save_path=None, downsample=None, target_pred=None):
    preds = calc_hair_color_from_images(hair_loss_class, tensor_images)
    image_grid = add_colors_to_images(preds, tensor_images, target_pred=target_pred)
    if downsample is not None:
        width, height = image_grid.size
        image_grid = transforms.Resize((width // downsample, height // downsample), interpolation=Image.BILINEAR)(image_grid)
    if save_path is not None:
        image_grid.save(save_path)
    return image_grid


if __name__ == '__main__':
    import argparse
    from gan_control.datasets.ffhq_dataset import get_ffhq_data_loader
    from gan_control.utils.file_utils import read_json
    from gan_control.losses.loss_model import LossModelClass

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--number_os_samples', type=int, default=5000)
    args = parser.parse_args()
    config = read_json(args.config_path, return_obj=True)
    loader = get_ffhq_data_loader(config.data_config, batch_size=args.batch_size, training=True, size=config.model_config['size'])
    age_loss_class = LossModelClass(config.training_config['hair_loss'], loss_name='hair_loss', mini_batch_size=args.batch_size, device="cuda")

    tensor_images, _ = next(loader)
    make_hair_seg_grid(age_loss_class, tensor_images[:8], nrow=4, save_path='path to save image')  # TODO:

