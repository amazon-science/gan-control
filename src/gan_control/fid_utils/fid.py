import argparse
import pickle

import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm

from gan_control.models.gan_model import Generator
from gan_control.fid_utils.calc_inception import load_patched_inception_v3


@torch.no_grad()
def extract_feature_from_samples(
        generator,
        inception,
        batch_size,
        n_sample,
        device="cuda",
        training=True):
    with torch.no_grad():
        n_batch = n_sample // batch_size
        resid = n_sample - (n_batch * batch_size)
        batch_sizes = [batch_size] * n_batch + [resid]
        if resid == 0:
            batch_sizes = [batch_size] * n_batch
        features = []

        for batch in tqdm(batch_sizes, disable=training):
            latent = torch.randn(batch, 512, device=device)
            img, _ = generator([latent])
            if img.shape[1] == 1:
                img = torch.cat([img[:,0,:,:].unsqueeze(dim=1), img[:,0,:,:].unsqueeze(dim=1), img[:,0,:,:].unsqueeze(dim=1)], dim=1)
            feat = inception(img)[0].view(img.shape[0], -1)
            features.append(feat.to('cpu'))

        features = torch.cat(features, 0)

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid

