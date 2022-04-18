# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms, utils
from PIL import Image

from gan_control.utils.ploting_utils import plot_hist
from gan_control.utils.hopenet_utils import softmax_temperature, draw_axis
from gan_control.utils.logging_utils import get_logger
from gan_control.utils.pil_images_utils import create_image_grid_from_image_list

_log = get_logger(__name__)


def calc_orientation_from_features(features):
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor)

    _, yaw_bpred = torch.max(features[:, 0, :], 1)
    _, pitch_bpred = torch.max(features[:, 1, :], 1)
    _, roll_bpred = torch.max(features[:, 2, :], 1)

    yaw_predicted = softmax_temperature(features[:, 0, :], 1)
    pitch_predicted = softmax_temperature(features[:, 1, :], 1)
    roll_predicted = softmax_temperature(features[:, 2, :], 1)

    yaw_predicted = torch.sum(yaw_predicted.cpu() * idx_tensor, 1) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.cpu() * idx_tensor, 1) * 3 - 99
    roll_predicted = torch.sum(roll_predicted.cpu() * idx_tensor, 1) * 3 - 99

    return yaw_predicted, pitch_predicted, roll_predicted


def calc_orientation_from_tensor_images(orientation_loss_class, tensor_images):
    with torch.no_grad():
        features_list = orientation_loss_class.calc_features(tensor_images)
    features = features_list[-1]
    yaw_predicted, pitch_predicted, roll_predicted = calc_orientation_from_features(features)
    return yaw_predicted, pitch_predicted, roll_predicted


def write_orientation_to_image(images, yaw_predicted, pitch_predicted, roll_predicted):
    pil_images_orientation = []
    for image_num in range(len(images)):
        pil_image = images[image_num]
        draw_axis(images[image_num], yaw_predicted[image_num], pitch_predicted[image_num], roll_predicted[image_num])
        pil_images_orientation.append(pil_image)
    return pil_images_orientation


def draw_orientation_to_tensor_images(orientation_loss_class, tensor_images):
    yaw_predicted, pitch_predicted, roll_predicted = calc_orientation_from_tensor_images(orientation_loss_class, tensor_images)
    tensor_images = tensor_images.mul(0.5).add(0.5).clamp(min=0., max=1.)
    pil_images_with_orientation = []
    for tensor_num in range(tensor_images.shape[0]):
        pil_image = transforms.ToPILImage()(tensor_images[tensor_num])
        draw_axis(pil_image, yaw_predicted[tensor_num], pitch_predicted[tensor_num], roll_predicted[tensor_num])
        pil_images_with_orientation.append(pil_image)
    return pil_images_with_orientation


def make_orientation_hist(orientation_loss_class, loader=None, generator=None, number_os_samples=2000, batch_size=40, title=None, save_path=None):
    total_yaw_predicted, total_pitch_predicted, total_roll_predicted = torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_num in tqdm(range(0, number_os_samples, batch_size)):
        if loader is not None:
            tensor_images, _ = next(loader)
            tensor_images = tensor_images.cuda()
        elif generator is not None:
            tensor_images = generator.gen_random()
        else:
            raise ValueError('loader and generator are None')
        yaw_predicted, pitch_predicted, roll_predicted = calc_orientation_from_tensor_images(orientation_loss_class, tensor_images)
        total_yaw_predicted = torch.cat([total_yaw_predicted, yaw_predicted], dim=0)
        total_pitch_predicted = torch.cat([total_pitch_predicted, pitch_predicted], dim=0)
        total_roll_predicted = torch.cat([total_roll_predicted, roll_predicted], dim=0)
    yaw = - (total_yaw_predicted.squeeze().numpy() * np.pi / 180)
    pitch = total_pitch_predicted.squeeze().numpy() * np.pi / 180
    roll = total_roll_predicted.squeeze().numpy() * np.pi / 180
    arrays = [yaw, pitch, roll]
    plot_hist(
        arrays,
        title=title,
        labels=['yaw', 'pitch', 'roll'],
        xlabel='Angles [radians]',
        bins=100,
        ncol=3,
        percentiles=(0.2, 0.5, 0.8),
        min_lim=-1000,
        max_lim=1000,
        save_path=save_path
    )
    return yaw, pitch, roll


def make_orientation_grid(orientation_loss_class, tensor_images, nrow=6, save_path=None, downsample=None):
    pil_images_with_orientation = draw_orientation_to_tensor_images(orientation_loss_class, tensor_images)
    image_grid = create_image_grid_from_image_list(pil_images_with_orientation, nrow=nrow)
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
    parser.add_argument('--config_path', type=str, default='../configs512/default_config_512.json')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--number_os_samples', type=int, default=2000)
    args = parser.parse_args()
    config = read_json(args.config_path, return_obj=True)
    if config.data_config['data_set_name'] == 'ffhq':
        from gan_control.datasets.ffhq_dataset import get_ffhq_data_loader
        loader = get_ffhq_data_loader(config.data_config, batch_size=args.batch_size, training=True, size=config.model_config['size'])
    elif config.data_config['data_set_name'] == 'afhq':
        from gan_control.datasets.afhq_dataset import get_afhq_data_loader
        loader = get_afhq_data_loader(config.data_config, batch_size=args.batch_size, training=True, size=config.model_config['size'])
    elif config.data_config['data_set_name'] == 'met-faces':
        from gan_control.datasets.metfaces_dataset import get_metfaces_data_loader
        loader = get_metfaces_data_loader(config.data_config, batch_size=args.batch_size, training=True, size=config.model_config['size'])

    # loader = get_ffhq_data_loader(config.data_config, batch_size=args.batch_size, training=True, size=config.model_config['size'])

    orientation_loss_model = None
    orientation_loss_model = LossModelClass(config.training_config['orientation_loss'], loss_name='orientation_loss', mini_batch_size=args.batch_size, device="cuda")

    make_orientation_hist(orientation_loss_model, loader=loader, number_os_samples=args.number_os_samples, batch_size=args.batch_size, title=None, save_path='path to save image')  # TODO:
    tensor_images, _ = next(loader)
    make_orientation_grid(orientation_loss_model, tensor_images[:8], nrow=4, save_path='path to save image')  # TODO:

