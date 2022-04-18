# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


class Face3dmmCriterion:

    def __init__(self, loss_name='defualt'):
        super(Face3dmmCriterion, self).__init__()
        self.loss_name = loss_name
        self.mse = torch.nn.MSELoss()

    def __call__(self, signatures: torch.Tensor, queries: torch.Tensor):
        #if self.loss_name in ['recon_3d_loss', 'gamma_loss', 'id_loss', 'ex_loss', 'tex_loss', 'angles_loss', 'xy_loss', 'z_loss']:
        signatures = signatures.unsqueeze(dim=1)
        queries = queries.unsqueeze(dim=0)
        diff = signatures - queries
        distances = torch.mean(torch.abs(diff), dim=-1)
        #distances = torch.sqrt(torch.pow(diff, 2).sum(dim=-1)) used in some evaluations
        return distances

    def controller_criterion(self, pred, target):
        return torch.abs(pred - target).mean()
