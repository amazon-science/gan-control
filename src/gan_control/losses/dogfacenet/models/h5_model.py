import torch
import os
import tensorflow as tf
import tensorflow.keras.backend as K

import numpy as np
from PIL import Image

alpha = 0.3

def triplet(y_true, y_pred):
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]

    ap = K.sum(K.square(a - p), -1)
    an = K.sum(K.square(a - n), -1)

    return K.sum(tf.nn.relu(ap - an + alpha))


def triplet_acc(y_true, y_pred):
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]

    ap = K.sum(K.square(a - p), -1)
    an = K.sum(K.square(a - n), -1)

    return K.less(ap + alpha, an)


class NeuralNetPB(object):
    def __init__(self, model_filepath):
        self.input = None
        self.graph = None
        self.placeholders = None
        self.layers = None
        self.weights = {}
        self.model_filepath = model_filepath
        self.batchsize = 10
        self._load_graph()

    def _load_graph(self):
        self.graph = tf.keras.models.load_model(
            self.model_filepath, custom_objects={'triplet':triplet, 'triplet_acc':triplet_acc}
        )
        # Get layer names
        self.layers = self.graph.layers
        # Get layer weights
        for i, layer in enumerate(self.layers):
            for weight0, weight1 in zip(layer.weights, layer.get_weights()):
                self.weights[os.path.join('%04d' % i, weight0.name)] = weight1
        self.weight_names = list(self.weights.keys())
        self.weight_names.sort()

    @staticmethod
    def convert_to_torch(weight, permute=False):
        weight_tensor = torch.FloatTensor(weight)
        if permute:
            #weight = np.transpose(weight)
            weight_tensor = weight_tensor.permute(3, 2, 0, 1)
        #weight_tensor = torch.FloatTensor(weight)
        return weight_tensor

    def convert_bn(self, pytorch_bn, start_key):
        pytorch_bn.weight[:] = self.convert_to_torch(self.weights[start_key + 'gamma:0'])
        pytorch_bn.bias[:] = self.convert_to_torch(self.weights[start_key + 'beta:0'])
        pytorch_bn.running_mean[:] = self.convert_to_torch(self.weights[start_key + 'moving_mean:0'])
        pytorch_bn.running_var[:] = self.convert_to_torch(self.weights[start_key + 'moving_variance:0'])

    def convert_conv(self, pytorch_conv, start_key):
        pytorch_conv.weight[:, :, :, :] = self.convert_to_torch(
            self.weights[start_key], permute=True
        )

    def convert_fc(self, pytorch_fc, start_key):
        pytorch_fc.weight[:, :] = self.convert_to_torch(self.weights[start_key]).permute(1, 0)

    def sanity_check(self, pytorch_net, random=False):
        pytorch_net.eval()
        if random:
            tf_input = np.random.randn(self.batchsize, 224, 224, 3) * 0 + 1
        else:
            pre_images = [Image.open('path to ffhq-dataset/images1024x1024/00000/%05d.png' % i) for i in range(10)] # TODO: path to example images
            images = [image.resize((224, 224), resample=Image.BICUBIC) for image in pre_images]
            images = [np.expand_dims(np.array(image)[:, :, ::-1], 0) for image in images]
            tf_input = np.concatenate(images, axis=0) / 255

        from torch.autograd import Variable
        pytorch_input = torch.FloatTensor(tf_input)
        pytorch_input = pytorch_input.permute(0, 3, 1, 2)
        for i, layer in enumerate(self.graph.layers):
            print('%d) %s' % (i, layer.name))
        pytorch_out = pytorch_net(pytorch_input, self.graph, tf_input)  #.detach.numpy()
        graph0 = tf.keras.Model(self.graph.input, self.graph.get_layer(index=48).output)
        tf_out = graph0.predict(tf_input)
        #tf_out = self.graph.predict(tf_input)

        temp = self.sess.run(self.layers, feed_dict={self.input: tf_input})
        lay_dict = {}
        for layer, fet in zip(self.layers, temp):
            lay_dict[layer] = fet

        print(lay_dict)


if __name__ == '__main__':
    from igt_res_gan.losses.dogfacenet.models.pytorch_dogfacenet_model import DogFaceNet
    neural_net_pb = NeuralNetPB(
        'path to keras_model_dogfacenet.249.h5'  # TODO: path to keras_model_dogfacenet.1.h5
    )
    pytorch_resnet = DogFaceNet()

    neural_net_pb.convert_conv(pytorch_resnet.conv0, neural_net_pb.weight_names[0])
    neural_net_pb.convert_bn(pytorch_resnet.bn0, os.path.split(neural_net_pb.weight_names[1])[0] + '/')

    neural_net_pb.convert_conv(pytorch_resnet.res_block1.conv0, neural_net_pb.weight_names[5])
    neural_net_pb.convert_bn(pytorch_resnet.res_block1.bn0, os.path.split(neural_net_pb.weight_names[6])[0] + '/')
    neural_net_pb.convert_conv(pytorch_resnet.res_block1.conv1, neural_net_pb.weight_names[10])
    neural_net_pb.convert_bn(pytorch_resnet.res_block1.bn1, os.path.split(neural_net_pb.weight_names[11])[0] + '/')
    neural_net_pb.convert_conv(pytorch_resnet.res_block1.conv2, neural_net_pb.weight_names[15])
    neural_net_pb.convert_bn(pytorch_resnet.res_block1.bn2, os.path.split(neural_net_pb.weight_names[16])[0] + '/')

    neural_net_pb.convert_conv(pytorch_resnet.res_block2.conv0, neural_net_pb.weight_names[20])
    neural_net_pb.convert_bn(pytorch_resnet.res_block2.bn0, os.path.split(neural_net_pb.weight_names[21])[0] + '/')
    neural_net_pb.convert_conv(pytorch_resnet.res_block2.conv1, neural_net_pb.weight_names[25])
    neural_net_pb.convert_bn(pytorch_resnet.res_block2.bn1, os.path.split(neural_net_pb.weight_names[26])[0] + '/')
    neural_net_pb.convert_conv(pytorch_resnet.res_block2.conv2, neural_net_pb.weight_names[30])
    neural_net_pb.convert_bn(pytorch_resnet.res_block2.bn2, os.path.split(neural_net_pb.weight_names[31])[0] + '/')

    neural_net_pb.convert_conv(pytorch_resnet.res_block3.conv0, neural_net_pb.weight_names[35])
    neural_net_pb.convert_bn(pytorch_resnet.res_block3.bn0, os.path.split(neural_net_pb.weight_names[36])[0] + '/')
    neural_net_pb.convert_conv(pytorch_resnet.res_block3.conv1, neural_net_pb.weight_names[40])
    neural_net_pb.convert_bn(pytorch_resnet.res_block3.bn1, os.path.split(neural_net_pb.weight_names[41])[0] + '/')
    neural_net_pb.convert_conv(pytorch_resnet.res_block3.conv2, neural_net_pb.weight_names[45])
    neural_net_pb.convert_bn(pytorch_resnet.res_block3.bn2, os.path.split(neural_net_pb.weight_names[46])[0] + '/')

    neural_net_pb.convert_conv(pytorch_resnet.res_block4.conv0, neural_net_pb.weight_names[50])
    neural_net_pb.convert_bn(pytorch_resnet.res_block4.bn0, os.path.split(neural_net_pb.weight_names[51])[0] + '/')
    neural_net_pb.convert_conv(pytorch_resnet.res_block4.conv1, neural_net_pb.weight_names[55])
    neural_net_pb.convert_bn(pytorch_resnet.res_block4.bn1, os.path.split(neural_net_pb.weight_names[56])[0] + '/')
    neural_net_pb.convert_conv(pytorch_resnet.res_block4.conv2, neural_net_pb.weight_names[60])
    neural_net_pb.convert_bn(pytorch_resnet.res_block4.bn2, os.path.split(neural_net_pb.weight_names[61])[0] + '/')

    neural_net_pb.convert_conv(pytorch_resnet.res_block5.conv0, neural_net_pb.weight_names[65])
    neural_net_pb.convert_bn(pytorch_resnet.res_block5.bn0, os.path.split(neural_net_pb.weight_names[66])[0] + '/')
    neural_net_pb.convert_conv(pytorch_resnet.res_block5.conv1, neural_net_pb.weight_names[70])
    neural_net_pb.convert_bn(pytorch_resnet.res_block5.bn1, os.path.split(neural_net_pb.weight_names[71])[0] + '/')
    neural_net_pb.convert_conv(pytorch_resnet.res_block5.conv2, neural_net_pb.weight_names[75])
    neural_net_pb.convert_bn(pytorch_resnet.res_block5.bn2, os.path.split(neural_net_pb.weight_names[76])[0] + '/')

    neural_net_pb.convert_fc(pytorch_resnet.fc, neural_net_pb.weight_names[80])

    pytorch_resnet.eval()
    path = 'path to save python model'  # TODO: path to save python model
    torch.save(pytorch_resnet.state_dict(), path)
    pytorch_resnet_loaded = DogFaceNet()
    pytorch_resnet_loaded.load_state_dict(torch.load(path))
    neural_net_pb.sanity_check(pytorch_resnet_loaded)


    print(pytorch_resnet)