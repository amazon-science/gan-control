# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms, utils
from PIL import Image

from gan_control.utils.ploting_utils import plot_bar
from gan_control.utils.hopenet_utils import softmax_temperature, draw_axis
from gan_control.utils.logging_utils import get_logger
from gan_control.utils.pil_images_utils import create_image_grid_from_image_list, write_text_to_image

_log = get_logger(__name__)


def get_class(idx):
    classes = {
        0: 'Neutral',
        1: 'Happy',
        2: 'Sad',
        3: 'Surprise',
        4: 'Fear',
        5: 'Disgust',
        6: 'Anger',
        7: 'Contempt'}

    return classes[idx]


def calc_expression_from_features(features):
    batch_size, ensemble_size, emotion_num = features.shape
    emotion_votes = np.zeros((batch_size, emotion_num))
    emotion = [features[:, i, :].cpu().detach().numpy() for i in range(ensemble_size)]
    batches = range(batch_size)
    for e in emotion:
        e_idx = np.argmax(e, 1)
        emotion_votes[batches, e_idx] += 1
    return np.argmax(emotion_votes, 1)


def calc_expression_from_tensor_images(expression_loss_class, tensor_images):
    with torch.no_grad():
        features_list = expression_loss_class.calc_features(tensor_images)
    features = features_list[-1]
    expressions = calc_expression_from_features(features)
    return expressions


def calc_and_write_expression_to_image(expression_loss_class, tensor_images):
    expressions = calc_expression_from_tensor_images(expression_loss_class, tensor_images)
    tensor_images = tensor_images.mul(0.5).add(0.5).clamp(min=0., max=1.)
    images = [transforms.ToPILImage()(tensor_images[i]) for i in range(tensor_images.shape[0])]
    return write_expression_to_image(images, expressions)


def write_expression_to_image(images, expressions):
    pil_images_with_expressions = []
    for image_num in range(len(images)):
        pil_image = write_text_to_image(images[image_num], get_class(expressions[image_num]))
        pil_images_with_expressions.append(pil_image)
    return pil_images_with_expressions


def make_expression_bar(expression_loss_class, loader=None, generator=None, number_os_samples=2000, batch_size=40, title=None, save_path=None):
    total_expressions = np.array([])
    for batch_num in tqdm(range(0, number_os_samples, batch_size)):
        if loader is not None:
            tensor_images, _ = next(loader)
            tensor_images = tensor_images.cuda()
        elif generator is not None:
            tensor_images = generator.gen_random()
        else:
            raise ValueError('loader and generator are None')
        expressions = calc_expression_from_tensor_images(expression_loss_class, tensor_images)
        total_expressions = np.concatenate([total_expressions, expressions], axis=0)
    plot_bar(
        [total_expressions],
        [get_class(i) for i in range(8)],
        title=title,
        labels=None,
        save_path=save_path
    )
    return total_expressions


def make_expression_grid(expression_loss_class, tensor_images, nrow=6, save_path=None, downsample=None):
    pil_images_with_expressions = calc_and_write_expression_to_image(expression_loss_class, tensor_images)
    image_grid = create_image_grid_from_image_list(pil_images_with_expressions, nrow=nrow)
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
    parser.add_argument('--number_os_samples', type=int, default=70000)
    args = parser.parse_args()
    config = read_json(args.config_path, return_obj=True)
    loader = get_ffhq_data_loader(config.data_config, batch_size=args.batch_size, training=True, size=config.model_config['size'])
    expression_loss_class = None
    expression_loss_class = LossModelClass(config.training_config['expression_loss'], loss_name='expression_loss', mini_batch_size=args.batch_size, device="cuda")

    make_expression_bar(expression_loss_class, loader=loader, number_os_samples=args.number_os_samples, batch_size=args.batch_size, title=None, save_path='path to save')  # TODO:
    tensor_images, _ = next(loader)
    make_expression_grid(expression_loss_class, tensor_images[:8], nrow=4, save_path='path to save')  # TODO:

