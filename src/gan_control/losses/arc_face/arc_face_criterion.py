# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


class ArcFaceCriterion:
    """
    ArcFaceCriterion
        Implements an embedding distance
    """

    def __init__(self):
        super(ArcFaceCriterion, self).__init__()

    def __call__(self, signatures: torch.Tensor, queries: torch.Tensor):
        signatures = signatures.unsqueeze(dim=1)
        queries = queries.unsqueeze(dim=0)
        diff = signatures - queries
        distances = torch.sum(torch.pow(diff, 2), dim=-1)
        #print(distances.shape)
        return distances