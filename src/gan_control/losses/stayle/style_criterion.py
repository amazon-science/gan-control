# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


class StyleCriterion:
    def __init__(self):
        super(StyleCriterion, self).__init__()

    def __call__(self, signatures: torch.Tensor, queries: torch.Tensor):
        signatures = signatures.unsqueeze(dim=1).unsqueeze(dim=1)
        queries = queries.unsqueeze(dim=1).unsqueeze(dim=0)
        diff = signatures - queries
        distances = torch.mean(torch.pow(diff, 2), dim=(-2, -1))
        return distances.squeeze(dim=2) * 1e5
