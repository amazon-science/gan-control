# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import cv2
import face_alignment
from skimage import io
import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch
from torchvision import utils, transforms


def make_68_ln_to_5_lm(Lm3D):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(Lm3D[lm_idx[[3, 4]], :], 0),
                     Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D


def load_lm3d():
    path = 'path_to_similarity_Lm3D_all.mat'
    assert path != 'path_to_similarity_Lm3D_all.mat', 'download similarity_Lm3D_all.mat from https://github.com/microsoft/Deep3DFaceReconstruction/blob/master/BFM/similarity_Lm3D_all.mat'
    Lm3D = loadmat('path')
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    return make_68_ln_to_5_lm(Lm3D)


#calculating least square problem
def POS(xp,x):
    npts = xp.shape[1]

    A = np.zeros([2*npts,8])

    A[0:2*npts-1:2,0:3] = x.transpose()
    A[0:2*npts-1:2,3] = 1

    A[1:2*npts:2,4:7] = x.transpose()
    A[1:2*npts:2,7] = 1;

    b = np.reshape(xp.transpose(),[2*npts,1])

    k,_,_,_ = np.linalg.lstsq(A,b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    t = np.stack([sTx,sTy],axis = 0)

    return t,s


def process_img(img,lm,t,s,target_size = 224.):
    w0,h0 = img.size
    w = (w0/s*102).astype(np.int32)
    h = (h0/s*102).astype(np.int32)
    img = img.resize((w,h),resample = Image.BICUBIC)

    left = (w/2 - target_size/2 + float((t[0] - w0/2)*102/s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*102/s)).astype(np.int32)
    below = up + target_size

    img = img.crop((left,up,right,below))
    img = np.array(img)
    #img = img[:,:,::-1] #RGBtoBGR
    img = np.expand_dims(img,0)
    lm = np.stack([lm[:,0] - t[0] + w0/2,lm[:,1] - t[1] + h0/2],axis = 1)/s*102
    lm = lm - np.reshape(np.array([(w/2 - target_size/2),(h/2-target_size/2)]),[1,2])

    return img,lm


# resize and crop input images before sending to the R-Net
def Preprocess(img,lm,lm3D,crop_size=224):

    w0,h0 = img.size

    # change from image plane coordinates to 3D sapce coordinates(X-Y plane)
    lm = np.stack([lm[:,0],h0 - 1 - lm[:,1]], axis = 1)

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t,s = POS(lm.transpose(),lm3D.transpose())

    # processing the image
    img_new,lm_new = process_img(img,lm,t,s,target_size=crop_size)
    lm_new = np.stack([lm_new[:,0],223 - lm_new[:,1]], axis = 1)
    trans_params = np.array([w0,h0,102.0/s,t[0],t[1]])

    return img_new,lm_new,trans_params


def align_face_by_image_path(path, fa=None, size=None, lm3D=None):
    input = io.imread(path)
    preds, source_image, algin_image = align_face_by_image(input, fa=fa, size=size, lm3D=lm3D)
    return preds, source_image, algin_image


def align_face_by_image(input, fa=None, size=None, lm3D=None, crop_size=224):
    input_is_tensor = False
    if isinstance(input, torch.Tensor):
        input_is_tensor = True
        if input.min() < 0:
            input = input.mul(0.5).add(0.5).clamp(min=0., max=1.) * 255
        input = input.numpy().astype('uint8')
        input = input.swapaxes(0, 1).swapaxes(1, 2)
    if fa is None:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
    if size is not None:
        input = cv2.resize(input, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
    if lm3D is None:
        lm3D = load_lm3d()
    with torch.no_grad():
        try:
            preds = fa.get_landmarks(input)
        except:
            preds = None
    if preds is not None:
        img_new, lm_new, trans_params = Preprocess(Image.fromarray(input), make_68_ln_to_5_lm(preds[0]), lm3D, crop_size=crop_size)
        img_new = img_new[0]
    else:
        img_new = cv2.resize(input, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    if input_is_tensor:
        img_new = img_new.swapaxes(1, 2).swapaxes(0, 1)
        img_new = torch.tensor(img_new).float().div(255).add(-0.5).mul(2)
        img_new = img_new.unsqueeze(0)
    return preds, input, img_new


def align_tensor_images(tensor, fa=None, lm3D=None, crop_size=224):
    is_cuda = tensor.device.type == 'cuda'
    tensor_list = []
    for i in range(tensor.shape[0]):
        preds, input, align_image = align_face_by_image(tensor[i].cpu(), fa=fa, lm3D=lm3D, crop_size=crop_size)
        tensor_list.append(align_image)
    out_tensor = torch.cat(tensor_list, dim=0)
    if is_cuda:
        out_tensor = out_tensor.cuda()
    return out_tensor


def paint_pred_on_face(image, pred):
    for i in range(pred.shape[0]):
        image[int(pred[i, 1]), int(pred[i, 0]), 0] = 255
        image[int(pred[i, 1]), int(pred[i, 0]), 1] = 0
        image[int(pred[i, 1]), int(pred[i, 0]), 2] = 0
    return image

