# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch


def calc_vectors_mean_and_std(vecs_list, all_vs_all=True):
    vecs = np.array(vecs_list)
    mean_vecs = vecs.mean(axis=0)
    if not all_vs_all:
        distances = np.sqrt(np.sum(np.power(vecs - mean_vecs, 2), axis=1))
    else:
        distances_list = []
        sig = torch.tensor(vecs).unsqueeze(0)
        gue_chunks = torch.tensor(vecs).split(100, dim=0)
        for gue_chunk in gue_chunks:
            gue_chunk = gue_chunk.unsqueeze(1)
            distances = torch.pow(gue_chunk - sig, 2)
            distances = torch.sqrt(torch.sum(distances, dim=-1))
            distances_list.append(distances)
        distances = torch.cat(distances_list, dim=0)
        mask = np.tril(np.ones([len(vecs_list), len(vecs_list)]), -1) == 1
        distances = distances.numpy()[mask]

    return mean_vecs, distances.mean()
