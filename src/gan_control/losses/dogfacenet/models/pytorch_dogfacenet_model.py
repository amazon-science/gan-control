import numpy as np
import torch
import torch.nn as nn
#import tensorflow as tf


#def print_diff(name, py_x, tf_m, index, tf_input):
#    py_x = py_x.detach().numpy()
#    print('Layer: %s' % name)
#    graph0 = tf.keras.Model(tf_m.input, tf_m.get_layer(index=index).output)
#    tf_x = graph0.predict(tf_input)
#    print('shapes: py %s, tf %s' % (str(py_x.shape), str(tf_x.shape)))
#    if len(py_x.shape) == 4:
#        print('2x4: \npy %s, \ntf %s' % (str(py_x[0, 0, :4, :4]), str(tf_x[0, :4, :4, 0])))
#        print('diff: %s' % str(np.mean(np.abs(py_x[0, 0, :, :] - tf_x[0, :, :, 0]))))
#    else:
#        print('1: \npy %s, \ntf %s' % (str(py_x[0, :1]), str(tf_x[0, :1])))
#        print('diff: %s' % str(np.mean(np.abs(py_x[0, :] - tf_x[0, :]))))


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class ResBlock(nn.Module):
    def __init__(self, in_s, out_s, pad='reg'):
        super(ResBlock, self).__init__()
        if pad=='reg':
            self.pad0 = nn.ZeroPad2d((1, 1, 1, 1))
        else:
            self.pad0 = nn.ZeroPad2d((0, 1, 0, 1))
        self.conv0 = nn.Conv2d(in_s, out_s, kernel_size=3, stride=(2,2), padding=0, bias=False)
        self.relu = nn.ReLU()
        self.bn0 = nn.BatchNorm2d(out_s)
        self.conv1 = nn.Conv2d(out_s, out_s, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_s)
        self.conv2 = nn.Conv2d(out_s, out_s, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_s)

    #def forward2(self, x, tf_model, tf_in):
    #    x = self.pad0(x)
    #    x = self.conv0(x)
    #    x = self.relu(x)
    #    print_diff('block3_conv0', x, tf_model, 20, tf_in)
    #    r = self.bn0(x)
    #    print_diff('block3_bn0', r, tf_model, 21, tf_in)
#
    #    x = self.conv1(r)
    #    x = self.relu(x)
    #    print_diff('block3_conv1', x, tf_model, 22, tf_in)
    #    x = self.bn1(x)
#
    #    r = r + x
    #    print_diff('block3_add0', r, tf_model, 24, tf_in)
    #    x = self.conv2(r)
    #    x = self.relu(x)
    #    x = self.bn2(x)
#
    #    r = r + x
    #    return r

    def forward(self, x):
        x = self.pad0(x)
        x = self.conv0(x)
        x = self.relu(x)
        r = self.bn0(x)

        x = self.conv1(r)
        x = self.relu(x)
        x = self.bn1(x)

        r = r + x

        x = self.conv2(r)
        x = self.relu(x)
        x = self.bn2(x)

        r = r + x
        return r


class DogFaceNet(nn.Module):
    def __init__(self):
        super(DogFaceNet, self).__init__()
        self.pad0 = nn.ZeroPad2d((2, 4, 2, 4))
        self.conv0 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.bn0 = nn.BatchNorm2d(16)
        self.maxpooling3 = nn.MaxPool2d(kernel_size=3)

        self.res_block1 = ResBlock(16,16)
        self.res_block2 = ResBlock(16, 32)
        self.res_block3 = ResBlock(32, 64, pad='b3')
        self.res_block4 = ResBlock(64, 128)
        self.res_block5 = ResBlock(128, 512)

        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 32, bias=False)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.pad0(x)
        x = self.conv0(x)
        x = self.relu(x)
        x = self.bn0(x)
        x = self.maxpooling3(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)

        x = self.global_avg_pooling(x)

        x = x.squeeze(3).squeeze(2)
        x = self.fc(x)
        x = l2_norm(x)
        return x

    #def forward2(self, x, tf_model, tf_in):
    #    x = self.pad0(x)
    #    x = self.conv0(x)
    #    x = self.relu(x)
    #    print_diff('conv0', x, tf_model, 1, tf_in)
    #    x = self.bn0(x)
    #    print_diff('bn0', x, tf_model, 2, tf_in)
    #    x = self.maxpooling3(x)
    #    print_diff('maxpool3', x, tf_model, 3, tf_in)
#
    #    x = self.res_block1(x)
    #    print_diff('block1', x, tf_model, 11, tf_in)
    #    x = self.res_block2(x)
    #    print_diff('block2', x, tf_model, 19, tf_in)
    #    x = self.res_block3.forward2(x, tf_model, tf_in)
    #    print_diff('block3', x, tf_model, 27, tf_in)
    #    x = self.res_block4(x)
    #    print_diff('block4', x, tf_model, 35, tf_in)
    #    x = self.res_block5(x)
    #    print_diff('block5', x, tf_model, 43, tf_in)
#
    #    x = self.global_avg_pooling(x)
#
    #    x = x.squeeze(3).squeeze(2)
    #    x = self.fc(x)
    #    x = l2_norm(x)
    #    print_diff('l2', x, tf_model, 48, tf_in)
    #    return x
#


