# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from gan_control.evaluation.hair import make_hair_color_grid


class HairCriterion:
    def __init__(self):
        super(HairCriterion, self).__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda()
        self.mse = torch.nn.MSELoss().cuda()

    def __call__(self, signatures: torch.Tensor, queries: torch.Tensor):
        b, c, h, w = signatures.shape
        thres = 0.01 * w*h
        signatures_masked_image = signatures[:, :3, :, :]
        signatures_mask = signatures[:, 3:, :, :]
        queries_masked_image = queries[:, :3, :, :]
        queries_mask = queries[:, 3:, :, :]

        signatures_mask_sum = torch.sum(signatures_mask.detach(), dim=[-2, -1])
        queries_mask_sum = torch.sum(queries_mask.detach(), dim=[-2, -1])
        signatures_mask_u = (signatures_mask_sum > thres).unsqueeze(dim=1)
        queries_mask_v = (queries_mask_sum > thres).unsqueeze(dim=0)
        valid_uv_mask = signatures_mask_u * queries_mask_v
        #valid_mask = (signatures_mask_sum > thres) * (queries_mask_sum > thres)

        signatures = torch.div(torch.sum(signatures_masked_image, dim=[-2, -1]), signatures_mask_sum + (signatures_mask_sum < 0.5).float())
        queries = torch.div(torch.sum(queries_masked_image, dim=[-2, -1]), queries_mask_sum + (queries_mask_sum < 0.5).float())
        # signatures = (signatures * self.std + self.mean)[valid_mask]
        # queries = (queries * self.std + self.mean)[valid_mask]
        signatures = signatures.mul(0.5).add(0.5) #* valid_mask  # signatures[valid_mask.squeeze()]
        queries = queries.mul(0.5).add(0.5) #* valid_mask  # queries[valid_mask.squeeze()]

        signatures = signatures.unsqueeze(dim=1)
        queries = queries.unsqueeze(dim=0)
        diff = signatures - queries
        diff = diff * valid_uv_mask
        distances = torch.mean(torch.abs(diff), dim=-1)

        return distances

    @staticmethod
    def predict(features):
        masked_image = features[:, :3, :, :]
        mask = features[:, 3:, :, :]
        mask_sum = torch.sum(mask.detach(), dim=[-2, -1])
        valid_mask = (mask_sum > 0.5)
        preds = torch.div(torch.sum(masked_image, dim=[-2, -1]), mask_sum + (mask_sum < 0.5).float())
        preds = preds.mul(0.5).add(0.5) * valid_mask
        return preds

    def controller_criterion(self, pred, target):
        return self.mse(pred, target)

    @staticmethod
    def visual(hair_loss_class, tensor_images, save_path=None, target_pred=None, nrow=4):
        return make_hair_color_grid(hair_loss_class, tensor_images, nrow=nrow, save_path=save_path, downsample=None, target_pred=target_pred)