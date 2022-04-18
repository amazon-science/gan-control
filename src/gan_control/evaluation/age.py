# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms, utils
from PIL import Image

from gan_control.utils.ploting_utils import plot_bar, plot_hist
from gan_control.utils.hopenet_utils import softmax_temperature, draw_axis
from gan_control.utils.logging_utils import get_logger
from gan_control.utils.pil_images_utils import create_image_grid_from_image_list, write_text_to_image

_log = get_logger(__name__)


def calc_age_from_tensor_images(age_loss_class, tensor_images):
    with torch.no_grad():
        features_list = age_loss_class.calc_features(tensor_images)
    features = features_list[-1]
    ages = age_loss_class.last_layer_criterion.get_predict_age(features.cpu())
    return ages


def calc_and_write_age_to_image(age_loss_class, tensor_images):
    ages = calc_age_from_tensor_images(age_loss_class, tensor_images)
    tensor_images = tensor_images.mul(0.5).add(0.5).clamp(min=0., max=1.)
    images = [transforms.ToPILImage()(tensor_images[i]) for i in range(tensor_images.shape[0])]
    return write_age_to_image(images, ages)


def write_age_to_image(images, ages):
    pil_images_with_ages = []
    for image_num in range(len(images)):
        pil_image = write_text_to_image(images[image_num], 'age: %.2f' % (ages[image_num]), place=(10, 50))
        pil_images_with_ages.append(pil_image)
    return pil_images_with_ages


def make_age_hist(age_loss_class, loader=None, generator=None, number_os_samples=2000, batch_size=40, title=None, save_path=None):
    total_ages = np.array([])
    for batch_num in tqdm(range(0, number_os_samples, batch_size)):
        if loader is not None:
            tensor_images, _ = next(loader)
            tensor_images = tensor_images.cuda()
        elif generator is not None:
            tensor_images = generator.gen_random()
        else:
            raise ValueError('loader and generator are None')
        ages = calc_age_from_tensor_images(age_loss_class, tensor_images)
        total_ages = np.concatenate([total_ages, ages], axis=0)
    plot_hist(
        [total_ages],
        title=title,
        labels=None,
        bins=151,
        plt_range=(0, 150),
        save_path=save_path
    )
    return total_ages


def make_ages_grid(ages_loss_class, tensor_images, nrow=6, save_path=None, downsample=None):
    pil_images_with_ages = calc_and_write_age_to_image(ages_loss_class, tensor_images)
    image_grid = create_image_grid_from_image_list(pil_images_with_ages, nrow=nrow)
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
    from gan_control.utils.ploting_utils import plot_hist
    from gan_control.losses.loss_model import LossModelClass

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--number_os_samples', type=int, default=5000)
    args = parser.parse_args()
    config = read_json(args.config_path, return_obj=True)
    loader = get_ffhq_data_loader(config.data_config, batch_size=args.batch_size, training=True, size=config.model_config['size'])
    age_loss_class = LossModelClass(config.training_config['age_loss'], loss_name='age_loss', mini_batch_size=args.batch_size, device="cuda")

    make_age_hist(age_loss_class, loader=loader, number_os_samples=args.number_os_samples, batch_size=args.batch_size, title=None, save_path='path to save image')  # TODO:
    tensor_images, _ = next(loader)
    make_ages_grid(age_loss_class, tensor_images[:8], nrow=4, save_path='path to save image')  # TODO:

