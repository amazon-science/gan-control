# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms, utils

from gan_control.utils.ploting_utils import plot_hist

from gan_control.utils.logging_utils import get_logger

_log = get_logger(__name__)


def re_arrange_inject_noise(noises):
    for i in range(0, noises[0].shape[0], 2):
        for j in range(len(noises)):
            noises[j][i + 1, :, :, :] = noises[j][i, :, :, :].detach()
    return noises


def compute_half_same_ids_embeddings_from_generator(
        generator,
        loss_model_class,
        num_of_samples,
        same_noise_for_same_id=False,
        return_images=False,
        same_chunk=(256, 512)
):
    images_list = []
    inject_noise = None
    latents = torch.randn([num_of_samples, 512])
    #start_list = list(range(same_chunk[0]))
    #end_list = list(range(same_chunk[1], 512))
    same_id_chunk = slice(same_chunk[0], same_chunk[1], 1)
    #same_pose_chunks = start_list + end_list
    even_half = slice(0, num_of_samples, 2)
    odd_half = slice(1, num_of_samples, 2)
    latents[odd_half, same_id_chunk] = latents[even_half, same_id_chunk]

    #if same_latent_side == 'id':
    #    latents[odd_half, 256:] = latents[even_half, 256:]
    #elif same_latent_side == 'pose':
    #    latents[odd_half, :256] = latents[even_half, :256]
    #else:
    #    raise ValueError('same_latent_side is %s (not in [id, pose])' % same_latent_side)
    out_latents = latents.numpy()
    latents = latents.chunk(num_of_samples // 20, dim=0)
    with torch.no_grad():
        for i in tqdm(range(num_of_samples // len(latents[0]))):
            if same_noise_for_same_id:
                if isinstance(generator, torch.nn.DataParallel):
                    inject_noise = generator.module.make_noise(batch_size=latents[i].shape[0])
                else:
                    inject_noise = generator.make_noise(batch_size=latents[i].shape[0])
                inject_noise = re_arrange_inject_noise(inject_noise)
            fake_images, _ = generator([latents[i].cuda()], noise=inject_noise)
            if return_images:
                images_list += [transforms.ToPILImage()(fake_images[i].mul(0.5).add(0.5).clamp(min=0.,max=1.).cpu()) for i in range(fake_images.shape[0])]
            if i == 0:
                embeddings = loss_model_class.calc_features(fake_images)
                embeddings = [embeddings[n].cpu() for n in range(len(embeddings))]
            else:
                feture_list = loss_model_class.calc_features(fake_images)
                feture_list = [feture_list[n].cpu() for n in range(len(feture_list))]
                embeddings = [torch.cat([embeddings[j], feture_list[j]], dim=0) for j in range(len(embeddings))]
        _log.info('extracted %d embeddings' % embeddings[0].shape[0])
    embeddings = [torch.cat([embeddings[n][even_half], embeddings[n][odd_half]], dim=0) for n in range(len(embeddings))]
    if return_images:
        images_list = images_list[even_half] + images_list[odd_half]
    return embeddings, images_list


def calc_separability(
        generator,
        loss_model_class,
        same_noise_for_same_id=False,
        num_of_samples=2000,
        save_path=None,
        title=None,
        return_images=False,
        same_chunk=(256, 512),
        last_layer_separability_only=False
):
    embeddings, images_list = compute_half_same_ids_embeddings_from_generator(
        generator,
        loss_model_class,
        num_of_samples,
        same_noise_for_same_id=same_noise_for_same_id,
        return_images=return_images,
        same_chunk=same_chunk
    )
    signatures_embs = [embeddings[i][:embeddings[i].shape[0] // 2] for i in range(len(embeddings))]
    queries_embs = [embeddings[i][embeddings[i].shape[0] // 2:] for i in range(len(embeddings))]
    signature_pids = np.array(range(signatures_embs[0].shape[0]))
    queries_pids = np.array(range(queries_embs[0].shape[0]))
    same_not_same_list = loss_model_class.calc_same_not_same_list(signatures_embs, queries_embs, signature_pids, queries_pids, last_layer_separability_only=last_layer_separability_only)
    for i in range(len(same_not_same_list)):
        arrays = [
            same_not_same_list[i]['same'],
            same_not_same_list[i]['not_same'],
            same_not_same_list[i]['all_not_same']
        ]
        if title is not None:
            plot_title = '%s layer %d' % (title, i)
        plot_hist(
            arrays,
            title=plot_title,
            labels=['same', 'not_same_2nd_best', 'all_not_same'],
            xlabel='Distance',
            bins=100,
            ncol=3,
            percentiles=(0.2, 0.5, 0.8),
            min_lim=0,
            max_lim=1000,
            save_path='%s_layer_%d.jpg' % (save_path, i)
        )
    return same_not_same_list, images_list
