# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from gan_control.evaluation.orientation import softmax_temperature

def calc_orientation_from_features(features):
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda()

    _, yaw_bpred = torch.max(features[:, 0, :], 1)
    _, pitch_bpred = torch.max(features[:, 1, :], 1)
    _, roll_bpred = torch.max(features[:, 2, :], 1)

    yaw_predicted = softmax_temperature(features[:, 0, :], 1)
    pitch_predicted = softmax_temperature(features[:, 1, :], 1)
    roll_predicted = softmax_temperature(features[:, 2, :], 1)

    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
    roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

    return yaw_predicted, pitch_predicted, roll_predicted

class HopenetCriterion:
    """
    HopenetCriterion
        Implements an embedding distance
    """

    def __init__(self):
        super(HopenetCriterion, self).__init__()

    def __call__(self, signatures: torch.Tensor, queries: torch.Tensor):
        signatures = signatures.unsqueeze(dim=1)
        queries = queries.unsqueeze(dim=0)
        diff = signatures - queries
        distances = torch.mean(torch.abs(diff), dim=(-2, -1))
        return distances

    def predict(self, features):
        yaw_predicted, pitch_predicted, roll_predicted = calc_orientation_from_features(features)
        return torch.cat([yaw_predicted.unsqueeze(1), pitch_predicted.unsqueeze(1), roll_predicted.unsqueeze(1)], dim=1)

    def controller_criterion(self, pred, target):
        return torch.abs(pred - target).mean()

