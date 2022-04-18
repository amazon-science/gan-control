# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from gan_control.fid_utils.fid import extract_feature_from_samples, calc_fid
import numpy as np
import pickle
import time


def evaluate_fid(generator, inception, batch, n_sample, device, inception_stat_path, training=False):
    start_time = time.time()
    generator.eval()
    inception.eval()

    features = extract_feature_from_samples(
        generator,
        inception,
        batch,
        n_sample,
        device,
        training=training
    ).numpy()
    print(f'extracted {features.shape[0]} features')

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    with open(inception_stat_path, 'rb') as f:
        embeds = pickle.load(f)
        real_mean = embeds['mean']
        real_cov = embeds['cov']

    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
    print('fid: %.3f, time: %.3f (min)' % (fid, (time.time() - start_time) / 60))
    return fid