# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# this file borrows heavily from https://github.com/rosinality/stylegan2-pytorch

import math
import random
import functools
import operator
import numpy as np

import torch
from torch import nn, autograd
from torch.nn import functional as F

from gan_control.utils.logging_utils import get_logger

_log = get_logger(__name__)

FUSED = False  # True enables NVIDIA cuda kernels

if FUSED:
    raise NotImplementedError('No Fused cuda kernels implemented')
    # import fused cuda kernels here: FusedLeakyReLU, fused_leaky_relu, upfirdn2d
else:
    class FusedLeakyReLU(nn.Module):
        def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
            super().__init__()
            self.negative_slope = negative_slope
            self.bias = nn.Parameter(torch.zeros(channel))
            self.scale = scale

        def forward(self, input):
            out = input + self.bias[None,:,None,None]
            out = F.leaky_relu(out, negative_slope=self.negative_slope)
            return out * self.scale
            # return out * math.sqrt(2)


    def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
        return scale * F.leaky_relu(input + bias.view((1, -1) + (1,) * (len(input.shape) - 2)),
                                    negative_slope=negative_slope)

    from gan_control.models.pytorch_upfirdn2d import upfirdn2d_native

    def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
        out = upfirdn2d_native(
            input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
        )

        return out

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        #print('input EqualConv2d:' + str(input.shape))  # TODO: alon
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        #print('out EqualConv2d:' + str(input.shape))  # TODO: alon
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            conv_transpose=False,
            overwrite_padding=None
    ):
        super().__init__()
        if not conv_transpose:
            raise ValueError('conv_transpose is %s' % str(conv_transpose))
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.conv_transpose = conv_transpose
        self.overwrite_padding = overwrite_padding

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        if self.overwrite_padding is not None:
            _log.info('ModulatedConv2d: overwrite_padding from %d to 0, in channel %d out channel %d' % (self.padding, self.in_channel, self.out_channel))
            self.padding = 0


        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            if self.conv_transpose:
                weight = weight.transpose(1, 2).reshape(
                    batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
                )
                out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
                _, _, height, width = out.shape
                out = out.view(batch, self.out_channel, height, width)
                out = self.blur(out)
            else:
                weight = weight.reshape(  # B,OC,IC,H,W
                    batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size  # BxOC,IC,H,W
                )
                out = F.interpolate(input, scale_factor=2, mode='nearest')
                out = nn.ReflectionPad2d(1)(out)
                out = F.conv2d(out, weight, padding=0, stride=1, groups=batch)
                _, _, height, width = out.shape
                out = out.view(batch, self.out_channel, height, width)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
            conv_transpose=False,
            overwrite_padding=None,
            noise_mode='normal'
    ):
        super().__init__()
        self.overwrite_padding = overwrite_padding

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            conv_transpose=conv_transpose,
            overwrite_padding=overwrite_padding
        )

        self.noise_mode = noise_mode
        if self.noise_mode in ['normal', 'same_for_same_id']:
            self.noise = NoiseInjection()
        elif self.noise_mode == 'zeros':
            self.noise = ModulatedNoiseInjection(zeros=True)
        elif self.noise_mode == 'id_zeros':
            self.noise = ModulatedNoiseInjection(id_zeros=True)

        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], out_channels=3, conv_transpose=False, overwrite_negative_padding=None):
        super().__init__()
        self.overwrite_negative_padding = overwrite_negative_padding
        if upsample:
            self.upsample = Upsample(blur_kernel)

        if self.overwrite_negative_padding is not None:
            _log.info('ToRGB overwrite_negative_padding from 0 to %.4f' % self.overwrite_negative_padding)

        self.conv = ModulatedConv2d(in_channel, out_channels, 1, style_dim, demodulate=False, conv_transpose=conv_transpose)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            if self.overwrite_negative_padding is not None:
                skip = F.pad(skip, (self.overwrite_negative_padding, self.overwrite_negative_padding,self.overwrite_negative_padding,self.overwrite_negative_padding))

            out = out + skip

        return out


class VAE(nn.Module):
    def __init__(self, lr_mlp=0.01, bottleneck_size=256):
        super(VAE, self).__init__()

        self.shared_in = nn.Sequential(
            EqualLinear(512, 512, lr_mul=lr_mlp, activation='fused_lrelu'),
            EqualLinear(512, 512, lr_mul=lr_mlp, activation='fused_lrelu'),
            EqualLinear(512, 512, lr_mul=lr_mlp, activation='fused_lrelu'),
        )

        self.to_mu = EqualLinear(512, bottleneck_size, lr_mul=lr_mlp, activation='fused_lrelu')
        self.to_sigma = EqualLinear(512, bottleneck_size, lr_mul=lr_mlp, activation='fused_lrelu')
        self.to_sample = EqualLinear(bottleneck_size, 512, lr_mul=lr_mlp, activation='fused_lrelu')

        self.shared_out = nn.Sequential(
            EqualLinear(512, 512, lr_mul=lr_mlp, activation='fused_lrelu'),
            EqualLinear(512, 512, lr_mul=lr_mlp, activation='fused_lrelu'),
            EqualLinear(512, 512, lr_mul=lr_mlp, activation='fused_lrelu'),
        )

    def encode(self, x):
        h1 = self.shared_in(x)
        return self.to_mu(h1), self.to_sigma(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.to_sample(z)
        return torch.sigmoid(self.shared_out(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class DoubleFcStack(nn.Module):
    def __init__(self, stack_1, stack_2):
        super(DoubleFcStack, self).__init__()
        self.stack_1 = stack_1
        self.stack_2 = stack_2

    def forward(self, x):
        x_1, x_2 = x.chunk(2, dim=1)
        x_1, x_2 = self.stack_1(x_1), self.stack_2(x_2)
        return torch.cat((x_1, x_2), dim=1)


class MultiFcStack(nn.Module):
    def __init__(self, fc_dict, fc_config):
        super(MultiFcStack, self).__init__()
        self.fc_config = fc_config
        for group_name in fc_config.in_order_group_names:
            self.__setattr__(group_name, fc_dict[group_name])

    def forward(self, x):
        x_new = []
        for group_name in self.fc_config.in_order_group_names:
            x_new.append(self.__getattr__(group_name)(
                x[:, self.fc_config.groups[group_name]['latent_place'][0]:self.fc_config.groups[group_name]['latent_place'][1]])
            )
        return torch.cat(x_new, dim=1)


class Generator(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            n_mlp,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,
            out_channels=3,
            vae=False,
            bottleneck_size=256,
            split_fc=False,
            marge_fc=False,
            fc_config=None,
            conv_transpose=False,
            model_mode='normal',
            noise_mode='normal'  # choices=['zeros', 'id_zeros', 'same_for_same_id']
    ):
        super().__init__()
        self.noise_mode = noise_mode
        self.model_mode = model_mode
        self.size = size
        self.vae = vae
        self.out_channels = out_channels
        self.fc_config = fc_config

        self.style_dim = style_dim

        if not vae:
            if not split_fc and not marge_fc:
                _log.info('using regular style FC stack')
                self.style = self.create_regular_fc_stack(lr_mlp, n_mlp, style_dim)
            elif split_fc:
                _log.info('using split style FC stack')
                self.style = self.make_fc_stacks_using_fc_config(fc_config, lr_mlp, n_mlp)
                # self.style = DoubleFcStack(layers_pose, layers_id)
            else:
                _log.info('using marge style FC stack')
                style0 = self.make_fc_stacks_using_fc_config(fc_config, lr_mlp, int(np.ceil(n_mlp / 2)))
                style1 = self.create_regular_fc_stack(lr_mlp, int(np.floor(n_mlp / 2)), style_dim)
                fc_stacks = [style0, style1]
                self.style = nn.Sequential(*fc_stacks)
        else:
            _log.info('using vae style embedding')
            self.style = VAE(lr_mlp, bottleneck_size=bottleneck_size)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: int(256 * channel_multiplier),
            128: int(128 * channel_multiplier),
            256: int(64 * channel_multiplier),
            512: int(32 * channel_multiplier),
            1024: int(16 * channel_multiplier),
            1344: int(16 * channel_multiplier),
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel, conv_transpose=conv_transpose, noise_mode=self.noise_mode
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False, out_channels=out_channels, conv_transpose=conv_transpose)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    conv_transpose=conv_transpose,
                    noise_mode=self.noise_mode
                )
            )
            overwrite_negative_padding = None
            overwrite_padding = None
            if model_mode == '896' and (2 ** i) == 16:
                overwrite_padding = 0
                overwrite_negative_padding = -1
            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel, conv_transpose=conv_transpose, overwrite_padding=overwrite_padding
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim, out_channels=out_channels, conv_transpose=conv_transpose, overwrite_negative_padding=overwrite_negative_padding))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2
        #self.conv_net = ConvNet(self.input, self.conv1, self.to_rgb1, self.to_rgbs, self.convs)

    def make_fc_stacks_using_fc_config(self, fc_config, lr_mlp, n_mlp):
        fc_list = {}
        for group_name in fc_config.in_order_group_names:
            _log.info('Adding %s to fc stacks, latent size %03d, latent place [%d:%d]' % (
                group_name,
                fc_config.groups[group_name]['latent_size'],
                fc_config.groups[group_name]['latent_place'][0],
                fc_config.groups[group_name]['latent_place'][1]
            ))
            new_fc_stack = self.create_fc_stack(lr_mlp, n_mlp, fc_config.groups[group_name]['latent_size'], mid_dim=256)
            fc_list[group_name] = new_fc_stack
        return MultiFcStack(fc_list, fc_config)


    def create_regular_fc_stack(self, lr_mlp, n_mlp, style_dim):
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        return nn.Sequential(*layers)


    def load_transfer_learning_model(self, transfer_learning_model, load_only_main=True):
        missing_keys, unexpected_keys = self.load_state_dict(transfer_learning_model.state_dict(), strict=False)
        if (len(missing_keys) > 0 or len(unexpected_keys) > 0) and not load_only_main:
            self.load_state_dict(transfer_learning_model.state_dict())
        else:
            for key in missing_keys:
                if key.split('.')[0] != 'style':
                    raise ValueError('missing key:%s is part of main network' % key)
            for key in unexpected_keys:
                if key.split('.')[0] != 'style':
                    raise ValueError('unexpected key:%s is part of main network' % key)
            _log.warning('Loading only main net found:\nmissing keys: %s\nunexpected keys: %s' % (str(missing_keys), str(unexpected_keys)))

    @staticmethod
    def create_fc_stack(lr_mlp, n_mlp, style_dim, mid_dim=None):
        mid_dim = mid_dim if mid_dim is not None else mid_dim
        layers = [PixelNorm()]
        for i in range(n_mlp):
            s_dim0 = style_dim
            s_dim1 = style_dim
            if i == 0:
                s_dim1 = mid_dim
            elif i < n_mlp - 1:
                s_dim0 = mid_dim
                s_dim1 = mid_dim
            elif i == n_mlp - 1:
                s_dim0 = mid_dim
            else:
                raise ValueError('debug')

            layers.append(
                EqualLinear(
                    s_dim0, s_dim1, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        return nn.Sequential(*layers)

    def make_noise(self, batch_size=1, device=None):
        if device is None:
            device = self.input.input.device

        noises = [torch.randn(batch_size, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for inter_layer in range(2):
                if self.model_mode == '896' and ((i > 4) or (i == 4 and inter_layer > 0)):
                    noises.append(torch.randn(batch_size, 1, 14 * (2 ** (i - 4)) , 14 * (2 ** (i - 4)), device=device))
                else:  # normal mode
                    noises.append(torch.randn(batch_size, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
            self,
            styles,
            return_latents=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
            return_grad=False
    ):
        #if noise is not None:
        #    print(noise[0].shape)
        #    print(noise[0])
        #    print(styles[0].shape)
        #    print(styles[0][:,506:])
        #    print('sen check')
        #    print(noise[0][0,0,0,0]*styles[0][:,510:])
        #else:
        #    print('noise is None: %s' % str(styles[0].device))
        if not input_is_latent:
            if not self.vae:
                styles = [self.style(s) for s in styles]
            else:
                vae_out = [self.style(s) for s in styles]
                styles, self.mu, self.logvar = list(map(list, zip(*vae_out)))

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)]


        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        #####
        #image = self.conv_net(latent, noise=noise)
        #if return_latents:
        #    return image, latent
        #else:
        #    return image, None
        #####
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip
        if return_grad:
            return image, self.g_path_regularize_grad(image, latent)

        elif return_latents:
            return image, latent

        else:
            return image, None

    @staticmethod
    def g_path_regularize_grad(fake_img, latents, dim_1_shape=1):
        noise = torch.randn_like(fake_img) / math.sqrt(
            fake_img.shape[2] * fake_img.shape[3] * dim_1_shape
        )
        grad, = autograd.grad(
            outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
        )
        return grad


class ConvNet(nn.Module):
    def __init__(self, input, conv1, to_rgb1, to_rgbs, convs):
        super().__init__()
        self.input = input
        self.conv1 = conv1
        self.to_rgb1 = to_rgb1
        self.to_rgbs = to_rgbs
        self.convs = convs

    def forward(self, latent, noise=None):
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip
        return image



class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], overwrite_padding=None):
        super().__init__()
        self.overwrite_padding = overwrite_padding
        if self.overwrite_padding is not None:
            _log.info('ResBlock: overwrite_padding is %.2f' % self.overwrite_padding)

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False,
        )

    def forward(self, input):
        if self.overwrite_padding is not None:
            input = F.pad(input, (int(self.overwrite_padding), int(self.overwrite_padding + 0.51), int(self.overwrite_padding), int(self.overwrite_padding + 0.51)))
        #print('input conv1:' + str(input.shape))  # TODO: alon
        out = self.conv1(input)
        #print('out conv1:' + str(out.shape))  # TODO: alon
        #print('input conv2:' + str(out.shape))  # TODO: alon
        out = self.conv2(out)
        #print('out conv2:' + str(out.shape))  # TODO: alon

        #print('input skip:' + str(input.shape))  # TODO: alon
        skip = self.skip(input)
        #print('out skip:' + str(skip.shape))  # TODO: alon
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], in_channels=3,
                 verification=False, verification_res_split=None, model_mode=None):
        super().__init__()
        self.model_mode = model_mode

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: int(256 * channel_multiplier),
            128: int(128 * channel_multiplier),
            256: int(64 * channel_multiplier),
            512: int(32 * channel_multiplier),
            1024: int(16 * channel_multiplier),
        }

        convs_shared = [ConvLayer(in_channels, channels[size], 1)]
        convs_adv = []
        convs_verification = []

        log_size = int(math.log(size, 2))

        self.verification = verification
        if verification_res_split is None:
            verification_res_split = int(size / 4)
        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            res = 2 ** (i - 1)
            out_channel = channels[res]
            overwrite_padding = None
            if self.model_mode == '896':
                if res == 32:
                    overwrite_padding = 1
                elif res == 16:
                    overwrite_padding = 1.5
            if verification and res < verification_res_split:
                convs_adv.append(ResBlock(in_channel, out_channel, blur_kernel, overwrite_padding=overwrite_padding))
                convs_verification.append(ResBlock(in_channel, out_channel, blur_kernel, overwrite_padding=overwrite_padding))
            else:
                convs_shared.append(ResBlock(in_channel, out_channel, blur_kernel, overwrite_padding=overwrite_padding))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs_shared)
        self.convs_adv = nn.Sequential(*convs_adv)
        self.convs_verification = nn.Sequential(*convs_verification)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )
        if verification:
            self.final_conv_verification = ConvLayer(in_channel + 1, channels[4], 3)
            self.final_linear_verification = nn.Sequential(
                EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
                EqualLinear(channels[4], 128),
            )

    def forward(self, input):
        out_shared = self.convs(input)
        out_adv = self.convs_adv(out_shared)
        if self.verification:
            out_ver = self.convs_verification(out_shared)

        out_adv = self._forward_split(out_adv, self.final_conv, self.final_linear)
        if self.verification:
            out_ver = self._forward_split(out_ver, self.final_conv_verification, self.final_linear_verification)
            return out_adv, out_ver
        else:
            return out_adv, None

    def _forward_split(self, out_shared, final_conv, final_linear):
        batch, channel, height, width = out_shared.shape
        group = min(batch, self.stddev_group)
        stddev = out_shared.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out_shared = torch.cat([out_shared, stddev], 1)
        out_shared = final_conv(out_shared)
        out_shared = out_shared.view(batch, -1)
        out_shared = final_linear(out_shared)
        return out_shared


class ModulatedNoiseInjection(nn.Module):
    def __init__(self, zeros=False, id_zeros=False):
        super().__init__()
        self.zeros = zeros
        self.id_zeros = id_zeros
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if self.zeros:
            return image
        elif self.id_zeros:
            if noise is None:
                batch, _, height, width = image.shape
                noise = image.new_empty(batch, 1, height, width).normal_()

            image_pose, image_id = torch.chunk(image, 2, dim=1)
            return torch.cat([image_pose + self.weight * noise, image_id], dim=1)


