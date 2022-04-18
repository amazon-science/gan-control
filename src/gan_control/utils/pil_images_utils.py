# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import subprocess as sp, shlex
from torchvision import transforms, utils
from PIL import Image, ImageDraw, ImageFont


def create_gif(save_dir, delay=50, name='sample'):
    sp.run(shlex.split(f'convert -delay {delay} -loop 0 -resize 3079x1027 *.jpg /mnt/md4/orville/Alon/gifs_8_6/{name}.gif'), cwd=save_dir)
    # sp.run('cd %s;cp sample.gif /mnt/md4/orville/Alon/gifs/%s.gif' % (save_dir, name))
    # sp.run(shlex.split(f'convert -delay {delay} -loop 0 *.jpg sample.gif'), cwd=save_dir)


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )


def create_image_grid_from_image_list(images, nrow=6):
    to_tensor = transforms.ToTensor()
    tensors = [to_tensor(images[i]).unsqueeze(0) for i in range(len(images))]
    tensors = torch.cat(tensors, dim=0)
    tensor_grid = utils.make_grid(tensors, nrow=nrow)
    return transforms.ToPILImage()(tensor_grid)


def write_text_to_image(image, text, size=36, place=(10, 10)):
    d = ImageDraw.Draw(image)
    d.text(place, text, fill=(255, 255, 0), font=ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', size))
    return image
