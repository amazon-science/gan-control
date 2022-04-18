# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import os
import pathlib
import sys
import time
from typing import List

import matplotlib.pyplot as plt
import streamlit as st
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision import utils
import numpy as np
import subprocess as sp, shlex
from sklearn.decomposition import PCA

# from orville_conditional_gan.image_generation.image_generator import save_image

from gan_control.inference.controller import Controller
from gan_control.projection.lpips.lpips import PerceptualLoss
from gan_control.utils.logging_utils import get_logger

_log = get_logger(__name__)


def tensor_to_numpy_img(tensor: torch.Tensor) -> np.array:
    tensor_tmp = tensor.mul(0.5).add(0.5).clamp(min=0., max=1.).cpu().detach()
    if tensor.ndimension == 3:
        tensor_tmp = tensor_tmp.premute(0,1,2)
    elif tensor.ndimension == 4:
        tensor_tmp = tensor_tmp.premute(0, 2, 3, 1)
    return tensor_tmp.numpy()


@torch.no_grad()
def get_pca_groups(controller, latent_mean, n_mean_latent, device):
    if isinstance(controller.model, nn.DataParallel):
        model = controller.model.module
    else:
        model = controller.model
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = model.style(noise_sample)
    latent_out = latent_out.detach().cpu().numpy()
    latent_out = latent_out - latent_mean.cpu().numpy()
    variance_percent = 0.5
    pca_weights = {}

    for group in controller.fc_controls.keys():
        if group == 'expression_q':
            continue
        pca = PCA()
        group_latent = controller.get_group_w_latent(latent_out, group)
        pca.fit(group_latent)
        idx_variance_percent = np.argmax(np.cumsum(pca.explained_variance_) / np.sum(pca.explained_variance_) > variance_percent)
        _log.info('%s PCA components: %s' % (group, str(idx_variance_percent)))
        pca_weight = pca.components_[:(idx_variance_percent+1), :]
        pca_weight = torch.from_numpy(pca_weight).cuda()
        pca_weights[group] = pca_weight
    return pca_weights


def plot_figures(lrs, mse_losses, n_losses, p_losses, axes=None):
    if axes is None:
        fig, axes = plt.subplots(2, 2)
        fig.tight_layout()
    axes[0, 0].plot(np.arange(len(p_losses)), p_losses)
    axes[0, 0].set_title('Perceptual Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 1].plot(np.arange(len(n_losses)), n_losses)
    axes[0, 1].set_title('Noise Loss')
    axes[0, 1].set_yscale('log')
    axes[1, 0].plot(np.arange(len(mse_losses)), mse_losses)
    axes[1, 0].set_title('MSE Loss')
    axes[1, 0].set_yscale('log')
    axes[1, 1].plot(np.arange(len(lrs)), lrs)
    axes[1, 1].set_title('Learning Rate')


def load_source_images(images: List[str], res=256):
    transform = transforms.Compose(
        [
            transforms.Resize(res),
            transforms.CenterCrop(res),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    source_tensors = []
    for image_name in images:
        loaded_image = Image.open(image_name).convert('RGB')
        source_tensors.append(transform(loaded_image).unsqueeze(dim=0))
    source_tensors = torch.cat(source_tensors, dim=0)
    return source_tensors


def merge_group_latents(controller, group, latent_in):
    group_latent = controller.get_group_w_latent(latent_in.data, group)
    controller.insert_group_w_latent(latent_in.data, group_latent.data.mean(dim=1).unsqueeze(1), group)


def get_avg_latent(model, n_mean_latent, device):
    if isinstance(model, nn.DataParallel):
        model = model.module

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = model.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
    latent_mean = latent_mean.cpu().numpy()
    latent_std = latent_std.cpu().numpy()
    return latent_mean, latent_std


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .type(torch.uint8)
            .permute(0, 2, 3, 1)
            .to("cpu")
            .numpy()
    )
