import torch
import torch.nn as nn
import numpy as np


def print_diff(name, py_x, tf_x):
    py_x = py_x.detach().numpy()
    print('Layer: %s' % name)
    print('shapes: py %s, tf %s' % (str(py_x.shape), str(tf_x.shape)))
    if len(py_x.shape) == 4:
        print('2x4: \npy %s, \ntf %s' % (str(py_x[0, 0, :2, :4]), str(tf_x[0, :2, :4, 0])))
        print('diff: %s' % str(np.mean(np.abs(py_x[0, 0, :, :] - tf_x[0, :, :, 0]))))
    else:
        print('1: \npy %s, \ntf %s' % (str(py_x[0, :1]), str(tf_x[0, :1])))
        print('diff: %s' % str(np.mean(np.abs(py_x[0, :] - tf_x[0, :]))))


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, eps=1.001e-5)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, eps=1.001e-5)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, eps=1.001e-5)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, eps=1.001e-5)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, eps=1.001e-5)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Block1(nn.Module):
    def __init__(self):
        super(Block1, self).__init__()
        self.unit_1 = StartBlock(64, 64, 256)
        self.unit_2 = MidBlock(256, 64, 256)
        self.unit_3 = EndBlock(256, 64, 256)

    def forward(self, x, ll):
        x = self.unit_1(x)
        x = self.unit_2(x)
        x = self.unit_3(x, ll, 1)
        return x

    def forward(self, x):
        x = self.unit_1(x)
        x = self.unit_2(x)
        x = self.unit_3(x)
        return x


class Block2(nn.Module):
    def __init__(self):
        super(Block2, self).__init__()
        self.unit_1 = StartBlock(256, 128, 512)
        self.unit_2 = MidBlock(512, 128, 512)
        self.unit_3 = MidBlock(512, 128, 512)
        self.unit_4 = EndBlock(512, 128, 512)

    def forward(self, x, ll):
        x = self.unit_1(x)
        x = self.unit_2(x)
        x = self.unit_3(x)
        x = self.unit_4(x, ll, 2)
        return x

    def forward(self, x):
        x = self.unit_1(x)
        x = self.unit_2(x)
        x = self.unit_3(x)
        x = self.unit_4(x)
        return x


class Block3(nn.Module):
    def __init__(self):
        super(Block3, self).__init__()
        self.unit_1 = StartBlock(512, 256, 1024)
        self.unit_2 = MidBlock(1024, 256, 1024)
        self.unit_3 = MidBlock(1024, 256, 1024)
        self.unit_4 = MidBlock(1024, 256, 1024)
        self.unit_5 = MidBlock(1024, 256, 1024)
        self.unit_6 = EndBlock(1024, 256, 1024)

    def forward(self, x, ll):
        x = self.unit_1(x)
        x = self.unit_2(x)
        x = self.unit_3(x)
        x = self.unit_4(x)
        x = self.unit_5(x)
        x = self.unit_6(x, ll, 3)
        return x

    def forward(self, x):
        x = self.unit_1(x)
        x = self.unit_2(x)
        x = self.unit_3(x)
        x = self.unit_4(x)
        x = self.unit_5(x)
        x = self.unit_6(x)
        return x


class Block4(nn.Module):
    def __init__(self):
        super(Block4, self).__init__()
        self.unit_1 = StartBlock(1024, 512, 2048)
        self.unit_2 = MidBlock(2048, 512, 2048)
        self.unit_3 = MidBlock(2048, 512, 2048)

    def forward(self, x):
        x = self.unit_1(x)
        x = self.unit_2(x)
        x = self.unit_3(x)
        return x


class StartBlock(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes):
        super(StartBlock, self).__init__()
        self.conv1 = conv1x1(in_planes, mid_planes)
        self.bn1 = nn.BatchNorm2d(mid_planes, eps=1.001e-5)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_planes, mid_planes)
        self.bn2 = nn.BatchNorm2d(mid_planes, eps=1.001e-5)
        self.conv3 = conv1x1(mid_planes, out_planes)
        self.bn3 = nn.BatchNorm2d(out_planes, eps=1.001e-5)

        self.conv_shortcut = conv1x1(in_planes, out_planes)
        self.bn_shortcut = nn.BatchNorm2d(out_planes, eps=1.001e-5)

    def forward(self, x):
        r = self.conv1(x)
        r = self.bn1(r)
        r = self.relu(r)
        r = self.conv2(r)
        r = self.bn2(r)
        r = self.relu(r)
        r = self.conv3(r)
        r = self.bn3(r)

        s = self.conv_shortcut(x)
        s = self.bn_shortcut(s)
        return self.relu(r + s)


class MidBlock(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes):
        super(MidBlock, self).__init__()
        self.conv1 = conv1x1(in_planes, mid_planes)
        self.bn1 = nn.BatchNorm2d(mid_planes, eps=1.001e-5)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_planes, mid_planes)
        self.bn2 = nn.BatchNorm2d(mid_planes, eps=1.001e-5)
        self.conv3 = conv1x1(mid_planes, out_planes)
        self.bn3 = nn.BatchNorm2d(out_planes, eps=1.001e-5)

    def forward(self, x):
        r = self.conv1(x)
        r = self.bn1(r)
        r = self.relu(r)
        r = self.conv2(r)
        r = self.bn2(r)
        r = self.relu(r)
        r = self.conv3(r)
        r = self.bn3(r)

        return self.relu(r + x)


class EndBlock(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes):
        super(EndBlock, self).__init__()
        self.conv1 = conv1x1(in_planes, mid_planes)
        self.bn1 = nn.BatchNorm2d(mid_planes, eps=1.001e-5)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_planes, mid_planes, stride=2, dilation=1)
        self.bn2 = nn.BatchNorm2d(mid_planes, eps=1.001e-5)
        self.conv3 = conv1x1(mid_planes, out_planes)
        self.bn3 = nn.BatchNorm2d(out_planes, eps=1.001e-5)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=2)
        self.unit_names = ['unit_3', 'unit_4', 'unit_6']

    def forward(self, x, ll, i):
        r = self.conv1(x)
        print_diff('end_block%d_conv1' % i, r, ll['import/resnet_v1_50/block%d/%s/bottleneck_v1/conv1/Conv2D:0' % (i, self.unit_names[i-1])])
        r = self.bn1(r)
        r = self.relu(r)
        r = self.conv2(r)
        print_diff('end_block%d_conv2' % i, r, ll['import/resnet_v1_50/block%d/%s/bottleneck_v1/conv2/Conv2D:0' % (i, self.unit_names[i-1])])
        r = self.bn2(r)
        r = self.relu(r)
        r = self.conv3(r)
        print_diff('end_block%d_conv3' % i, r, ll['import/resnet_v1_50/block%d/%s/bottleneck_v1/conv3/Conv2D:0' % (i, self.unit_names[i-1])])
        r = self.bn3(r)

        s = self.maxpool(x)
        print_diff('end_block%d_MaxPool' % i, s, ll['import/resnet_v1_50/block%d/%s/bottleneck_v1/shortcut/MaxPool:0' % (i, self.unit_names[i-1])])
        return self.relu(r + s)

    def forward(self, x):
        r = self.conv1(x)
        r = self.bn1(r)
        r = self.relu(r)
        r = self.conv2(r)
        r = self.bn2(r)
        r = self.relu(r)
        r = self.conv3(r)
        r = self.bn3(r)

        s = self.maxpool(x)
        return self.relu(r + s)


class TfFcBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(TfFcBlock, self).__init__()
        self.tf_fc = conv1x1(in_planes, out_planes)
        self.add_bais = torch.nn.parameter.Parameter(torch.ones([out_planes]))

    def forward(self, x):
        x = self.tf_fc(x)
        x = x.squeeze(2)
        x = x.squeeze(2)
        x = x + self.add_bais

        return x


class Recon3D(nn.Module):
    def __init__(self):
        super(Recon3D, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1.001e-5)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.block1 = Block1()
        self.block2 = Block2()
        self.block3 = Block3()
        self.block4 = Block4()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.id = TfFcBlock(2048, 80)
        self.ex = TfFcBlock(2048, 64)
        self.tex = TfFcBlock(2048, 80)
        self.angles = TfFcBlock(2048, 3)
        self.gamma = TfFcBlock(2048, 27)
        self.xy = TfFcBlock(2048, 2)
        self.z = TfFcBlock(2048, 1)

    def forward(self, x, ll):
        x = self.conv1(x)
        print_diff('conv1', x, ll['import/resnet_v1_50/conv1/Conv2D:0'])
        x = self.bn1(x)
        print_diff('bn1', x, ll['import/resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:0'])
        x = self.relu(x)
        print_diff('relu1', x, ll['import/resnet_v1_50/conv1/Relu:0'])
        # Added padding to correspond with TensorFlow
        x = self.maxpool(nn.functional.pad(x, (0, 1, 0, 1)))
        print_diff('maxpool1', x, ll['import/resnet_v1_50/pool1/MaxPool:0'])
        x = self.block1(x, ll)
        print_diff('block1', x, ll['import/resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0'])
        x = self.block2(x, ll)
        print_diff('block2', x, ll['import/resnet_v1_50/block2/unit_4/bottleneck_v1/Relu:0'])
        x = self.block3(x, ll)
        print_diff('block3', x, ll['import/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0'])
        x = self.block4(x)
        print_diff('block4', x, ll['import/resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0'])
        x = self.avgpool(x)
        print_diff('avgpool', x, ll['import/resnet_v1_50/pool5:0'])
        id = self.id(x)
        print_diff('id', id, ll['import/fc-id/squeezed:0'])
        ex = self.ex(x)
        print_diff('ex', ex, ll['import/fc-ex/squeezed:0'])
        tex = self.tex(x)
        print_diff('tex', tex, ll['import/fc-tex/squeezed:0'])
        angles = self.angles(x)
        print_diff('angles', angles, ll['import/fc-angles/squeezed:0'])
        gamma = self.gamma(x)
        print_diff('gamma', gamma, ll['import/fc-gamma/squeezed:0'])
        xy = self.xy(x)
        print_diff('xy', xy, ll['import/fc-XY/squeezed:0'])
        z = self.z(x)
        print_diff('z', z, ll['import/fc-Z/squeezed:0'])
        out = torch.cat([id, ex, tex, angles, gamma, xy, z], dim=1)
        print_diff('out', out, ll['import/coeff:0'])
        return out

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # Added padding to correspond with TensorFlow
        x = self.maxpool(nn.functional.pad(x, (0, 1, 0, 1)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        id = self.id(x)
        ex = self.ex(x)
        tex = self.tex(x)
        angles = self.angles(x)
        gamma = self.gamma(x)
        xy = self.xy(x)
        z = self.z(x)
        out = torch.cat([id, ex, tex, angles, gamma, xy, z], dim=1)
        return out


def _resnet(arch, block, layers, pretrained, **kwargs):
    model = Recon3D()

    return model


def resnet50(pretrained=False, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)