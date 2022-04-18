# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


class ImageNetCriterion:
    def __init__(self):
        super(ImageNetCriterion, self).__init__()

    def __call__(self, signatures: torch.Tensor, queries: torch.Tensor):
        signatures = signatures.unsqueeze(dim=1)
        queries = queries.unsqueeze(dim=0)
        diff = signatures - queries
        distances = torch.mean(torch.abs(diff), dim=(-1))
        #print(distances.shape)
        return distances