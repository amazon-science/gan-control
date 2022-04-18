# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

def center_crop_tensor(tensor, crop_size):
    b, c, h, w = tensor.shape
    up = (h - crop_size) // 2
    left = (w - crop_size) // 2
    tensor = tensor[:, :, up:up+crop_size, left:left+crop_size]
    return tensor
