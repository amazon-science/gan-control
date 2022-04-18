# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.nn import functional as F


class DeepAgeCriterion:
    """
    HopenetCriterion
        Implements an embedding distance
    """

    def __init__(self):
        super(DeepAgeCriterion, self).__init__()
        self.mse = torch.nn.MSELoss()

    def __call__(self, signatures: torch.Tensor, queries: torch.Tensor):
        signatures = signatures.unsqueeze(dim=1)
        queries = queries.unsqueeze(dim=0)
        diff = signatures - queries
        distances = torch.mean(torch.abs(diff), dim=(-1))
        return distances

    @staticmethod
    def get_predict_age(age_pb):
        predict_age_pb = F.softmax(age_pb, dim=-1)
        predict_age = torch.zeros(age_pb.size(0)).type_as(predict_age_pb)
        for i in range(age_pb.size(0)):
            for j in range(age_pb.size(1)):
                predict_age[i] += j * predict_age_pb[i][j]
        return predict_age

    def predict(self, age_pb):
        return self.get_predict_age(age_pb)

    def controller_criterion(self, pred, target):
        return self.mse(pred, target)