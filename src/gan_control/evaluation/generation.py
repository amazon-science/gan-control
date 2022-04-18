# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torchvision import transforms, utils
from PIL import Image

from gan_control.utils.logging_utils import get_logger

_log = get_logger(__name__)


def gen_grid(model, latent, injection_noise=None, nrow=4, downsample=None):
    with torch.no_grad():
        output_tensor, _ = model([latent], noise=injection_noise)
        image_tensor = output_tensor.mul(0.5).add(0.5).clamp(min=0., max=1.).cpu()
    image = transforms.ToPILImage()(utils.make_grid(image_tensor, nrow=nrow))
    if downsample is not None:
        width, height = image.size
        image = transforms.Resize((width // downsample, height // downsample), interpolation=Image.BILINEAR)(image)
    return image


def make_noise_id_pose_matrix(model, ids_in_row=6, pose_in_col=6, device='cpu', id_chunk=(256, 512)):
    ids = []
    poses = []
    noises = []
    z_sampels = []
    start_list = list(range(id_chunk[0]))
    end_list = list(range(id_chunk[1], 512))
    same_id_chunk = list(range(id_chunk[0], id_chunk[1]))
    same_pose_chunks = start_list + end_list
    for i in range(ids_in_row):
        sample_z = torch.randn(1, 512, device=device)
        ids.append(sample_z[0, same_id_chunk].unsqueeze(dim=0))
        poses.append(sample_z[0, same_pose_chunks].unsqueeze(dim=0))
    for row in range(pose_in_col):
        for col in range(ids_in_row):
            canvas = torch.zeros_like(torch.cat([poses[col], ids[row]], dim=1))
            canvas[:, same_id_chunk] = ids[row]
            canvas[:, same_pose_chunks] = poses[col]
            z_sampels.append(canvas)
    if isinstance(model, nn.DataParallel):
        noises = [model.module.make_noise(device=device) for _ in range(ids_in_row)]
    else:
        noises = [model.make_noise(device=device) for _ in range(ids_in_row)]
    return z_sampels, noises


@torch.no_grad()
def gen_matrix(
        model,
        ids_in_row=6,
        pose_in_col=6,
        latents=None,
        injection_noises=None,
        device='cuda',
        same_noise_per_id=False,
        downsample=None,
        return_list=False,
        same_chunk=(256, 512),
        same_noise_for_all=False
):
    if same_noise_per_id and same_noise_for_all:
        _log.warning('same_noise_for_all and same_noise_for_all is True -> same_noise_for_all')
    injection_noise = None
    injection_num = 0
    if latents is None or injection_noises is None:
        temp_latents, temp_injection_noises = make_noise_id_pose_matrix(model, ids_in_row=ids_in_row, pose_in_col=pose_in_col, device='cpu', id_chunk=same_chunk)
        if latents is None:
            latents = temp_latents
        if injection_noises is None:
            injection_noises = temp_injection_noises
    if same_noise_per_id or same_noise_for_all:
        injection_noise = injection_noises[injection_num]
        injection_noise = [injection_noise[n].cuda() for n in range(len(injection_noise))]
        injection_num += 1
    total_sample, _ = model([latents[0].cuda()], noise=injection_noise)
    for pic_num in range(1, ids_in_row * pose_in_col):
        if same_noise_per_id and (pic_num % ids_in_row == 0):
            injection_noise = injection_noises[pic_num]
            injection_noise = [injection_noise[n].cuda() for n in range(len(injection_noise))]
            injection_num += 1
        sample, _ = model([latents[pic_num].cuda()], noise=injection_noise)
        total_sample = torch.cat([total_sample, sample], dim=0)
    if return_list:
        return total_sample.cpu()
    total_sample = utils.make_grid(total_sample.mul(0.5).add(0.5).clamp(min=0., max=1.).cpu(), nrow=ids_in_row)
    image = transforms.ToPILImage()(total_sample)
    if downsample is not None:
        width, height = image.size
        image = transforms.Resize((width // downsample, height // downsample), interpolation=Image.BILINEAR)(image)
    return image


class IterableModel():
    def __init__(self, model, same_noise_for_same_id=False, batch_size=20):
        self.model = model
        self.same_noise_for_same_id=same_noise_for_same_id
        self.batch_size = batch_size

    def gen_random(self):
        random_latent = torch.randn(self.batch_size, 512, device='cuda')
        output, _ = self.model([random_latent.cuda()], noise=None)
        return output







