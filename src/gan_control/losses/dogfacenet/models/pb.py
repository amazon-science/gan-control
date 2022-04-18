import os
import tensorflow as tf
import torch
import numpy as np
from gan_control.losses.face3dmm_recon.models.pytorch_3d_recon_model import resnet50
from PIL import Image


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
            'path to keras_model_dogfacenet.1.h5'  # TODO: path to keras_model_dogfacenet.1.h5
        )
        # Get layer names
        self.layers = self.graph.layers
        # Get layer weights
        for i, layer in enumerate(self.layers):
            for weight0, weight1 in zip(layer.weights, layer.get_weights()):
                self.weights[os.path.join(str(i), weight0.name)] = tf.make_ndarray(weight1)

    def test(self, data, output_layer):
        """
        Feed-forward data and get the output from any layer.
        :param data: data to propagate.
        :param output_layer: the output layer (can be any hidden layer for debug).
        :return:
        """
        output_tensor = self.graph.get_tensor_by_name(output_layer)
        output = self.sess.run(output_tensor, feed_dict={self.input: data})
        return output

    @staticmethod
    def convert_to_torch(weight, permute=False):
        weight_tensor = torch.FloatTensor(weight)
        if permute:
            weight_tensor = weight_tensor.permute(3, 2, 0, 1)
        return weight_tensor

    def convert_bn(self, pytorch_bn, start_key):
        pytorch_bn.weight[:] = neural_net_pb.convert_to_torch(neural_net_pb.weights[start_key + 'gamma'])
        pytorch_bn.bias[:] = neural_net_pb.convert_to_torch(neural_net_pb.weights[start_key + 'beta'])
        pytorch_bn.running_mean[:] = neural_net_pb.convert_to_torch(neural_net_pb.weights[start_key + 'moving_mean'])
        pytorch_bn.running_var[:] = neural_net_pb.convert_to_torch(neural_net_pb.weights[start_key + 'moving_variance'])

    def convert_start_block(self, pytorch_unit, tf_block, tf_unit):
        pytorch_unit.conv1.weight[:, :, :, :] = self.convert_to_torch(self.weights['resnet_v1_50/%s/%s/bottleneck_v1/conv1/weights' % (tf_block, tf_unit)], permute=True)
        self.convert_bn(pytorch_unit.bn1, 'resnet_v1_50/%s/%s/bottleneck_v1/conv1/BatchNorm/' % (tf_block, tf_unit))

        pytorch_unit.conv2.weight[:, :, :, :] = self.convert_to_torch(self.weights['resnet_v1_50/%s/%s/bottleneck_v1/conv2/weights' % (tf_block, tf_unit)], permute=True)
        self.convert_bn(pytorch_unit.bn2, 'resnet_v1_50/%s/%s/bottleneck_v1/conv2/BatchNorm/' % (tf_block, tf_unit))

        pytorch_unit.conv3.weight[:, :, :, :] = self.convert_to_torch(self.weights['resnet_v1_50/%s/%s/bottleneck_v1/conv3/weights' % (tf_block, tf_unit)], permute=True)
        self.convert_bn(pytorch_unit.bn3, 'resnet_v1_50/%s/%s/bottleneck_v1/conv3/BatchNorm/' % (tf_block, tf_unit))

        pytorch_unit.conv_shortcut.weight[:, :, :, :] = self.convert_to_torch(self.weights['resnet_v1_50/%s/%s/bottleneck_v1/shortcut/weights' % (tf_block, tf_unit)], permute=True)
        self.convert_bn(pytorch_unit.bn_shortcut, 'resnet_v1_50/%s/%s/bottleneck_v1/shortcut/BatchNorm/' % (tf_block, tf_unit))

    def convert_mid_block(self, pytorch_unit, tf_block, tf_unit):
        pytorch_unit.conv1.weight[:, :, :, :] = self.convert_to_torch(self.weights['resnet_v1_50/%s/%s/bottleneck_v1/conv1/weights' % (tf_block, tf_unit)], permute=True)
        self.convert_bn(pytorch_unit.bn1, 'resnet_v1_50/%s/%s/bottleneck_v1/conv1/BatchNorm/' % (tf_block, tf_unit))

        pytorch_unit.conv2.weight[:, :, :, :] = self.convert_to_torch(self.weights['resnet_v1_50/%s/%s/bottleneck_v1/conv2/weights' % (tf_block, tf_unit)], permute=True)
        self.convert_bn(pytorch_unit.bn2, 'resnet_v1_50/%s/%s/bottleneck_v1/conv2/BatchNorm/' % (tf_block, tf_unit))

        pytorch_unit.conv3.weight[:, :, :, :] = self.convert_to_torch(self.weights['resnet_v1_50/%s/%s/bottleneck_v1/conv3/weights' % (tf_block, tf_unit)], permute=True)
        self.convert_bn(pytorch_unit.bn3, 'resnet_v1_50/%s/%s/bottleneck_v1/conv3/BatchNorm/' % (tf_block, tf_unit))

    def convert_end_block(self, pytorch_unit, tf_block, tf_unit):
        pytorch_unit.conv1.weight[:, :, :, :] = self.convert_to_torch(
            self.weights['resnet_v1_50/%s/%s/bottleneck_v1/conv1/weights' % (tf_block, tf_unit)], permute=True)
        self.convert_bn(pytorch_unit.bn1, 'resnet_v1_50/%s/%s/bottleneck_v1/conv1/BatchNorm/' % (tf_block, tf_unit))

        pytorch_unit.conv2.weight[:, :, :, :] = self.convert_to_torch(
            self.weights['resnet_v1_50/%s/%s/bottleneck_v1/conv2/weights' % (tf_block, tf_unit)], permute=True)
        self.convert_bn(pytorch_unit.bn2, 'resnet_v1_50/%s/%s/bottleneck_v1/conv2/BatchNorm/' % (tf_block, tf_unit))

        pytorch_unit.conv3.weight[:, :, :, :] = self.convert_to_torch(
            self.weights['resnet_v1_50/%s/%s/bottleneck_v1/conv3/weights' % (tf_block, tf_unit)], permute=True)
        self.convert_bn(pytorch_unit.bn3, 'resnet_v1_50/%s/%s/bottleneck_v1/conv3/BatchNorm/' % (tf_block, tf_unit))

    def sanity_check(self, pytorch_net, random=False):
        pytorch_net.eval()
        if random:
            tf_input = np.random.randn(self.batchsize,224,224,3)
        else:
            pre_images = [Image.open('path to ffhq-dataset/images1024x1024/00000/%05d.png' % i) for i in range(10)]  # TODO: path to example images
            images = [image.resize((224, 224),resample = Image.BICUBIC) for image in pre_images]
            images = [np.expand_dims(np.array(image)[:,:,::-1], 0) for image in images]
            tf_input = np.concatenate(images, axis=0)

        #pytorch_input = pytorch_input.permute(0, 3, 2, 1)
        pytorch_input = torch.FloatTensor(tf_input)
        pytorch_input = pytorch_input.permute(0, 3, 1, 2)

        #output_tensor = self.graph.get_tensor_by_name('import/coeff:0')
        #input_tensor = self.graph.get_tensor_by_name('import/resnet_v1_50/Pad:0')
        #conv1_tensor = self.graph.get_tensor_by_name('import/resnet_v1_50/conv1/Conv2D:0')
        #bn1_tensor = self.graph.get_tensor_by_name('import/resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:0')
        #relu1_tensor = self.graph.get_tensor_by_name('import/resnet_v1_50/conv1/Relu:0')
        #max1_tensor = self.graph.get_tensor_by_name('import/resnet_v1_50/pool1/MaxPool:0')
        #block1u1_tensor = self.graph.get_tensor_by_name('import/resnet_v1_50/block1/unit_1/bottleneck_v1/Relu:0')
        #block1u2_tensor = self.graph.get_tensor_by_name('import/resnet_v1_50/block1/unit_2/bottleneck_v1/Relu:0')
        #block1u3_tensor = self.graph.get_tensor_by_name('import/resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0')
        layers = []
        for i in range(len(self.layers)):
            if os.path.split(self.layers[i])[1] in ['Conv2D', 'Relu', 'FusedBatchNorm', 'MaxPool', 'Pad', 'Add', 'pool5', 'BiasAdd', 'concat', 'coeff', 'squeezed']:
                layers.append(self.layers[i] + ':0')
        temp = self.sess.run(layers, feed_dict={self.input: tf_input})
        lay_dict = {}
        for layer, fet in zip(layers, temp):
            lay_dict[layer] = fet
            if os.path.split(layer)[1] in ['Conv2D:0', 'Relu:0']:
                print('layer:%s' % layer)
                print(fet[0, 0, 0, 0])
        #print(tf_pad[0, :2, :3, 0])
        #print(tf_conv1[0, :2, :3, 0])
        #print(tf_bn1[0, :2, :3, 0])
        #print(tf_relu1[0, :2, :3, 0])
        #print(max1[0, :2, :3, 0])
        #print(block1u1[0, :2, :3, 0])
        #print(block1u2[0, :2, :3, 0])
        #print(block1u3[0, :2, :3, 0])


        pytorch_output = pytorch_net(pytorch_input, lay_dict).detach().numpy()

        #for i in range(tf_output.shape[0]):
        #    print(tf_output[i][100:104])
        #    print(pytorch_output[i][100:104])
        #    print(np.mean(tf_output[i] - pytorch_output[i]))
        #    print('\n')




if __name__ == '__main__':
    res_string = 'resnet_v1_50/'
    neural_net_pb = NeuralNetPB('path to keras_model_dogfacenet.1.h5')  # TODO: path to keras_model_dogfacenet.1.h5
    pytorch_resnet = DogFaceNet()

    pytorch_resnet.conv1.weight[:, :, :, :] = neural_net_pb.convert_to_torch(neural_net_pb.weights[res_string + 'conv1/weights'], permute=True)
    neural_net_pb.convert_bn(pytorch_resnet.bn1, res_string + 'conv1/BatchNorm/')
    # pytorch_resnet.bn1.weight[:] = neural_net_pb.convert_to_torch(neural_net_pb.weights[res_string + 'conv1/BatchNorm/gamma'])
    # pytorch_resnet.bn1.bias[:] = neural_net_pb.convert_to_torch(neural_net_pb.weights[res_string + 'conv1/BatchNorm/beta'])
    # pytorch_resnet.bn1.running_mean[:] = neural_net_pb.convert_to_torch(neural_net_pb.weights[res_string + 'conv1/BatchNorm/moving_mean'])
    # pytorch_resnet.bn1.running_var[:] = neural_net_pb.convert_to_torch(neural_net_pb.weights[res_string + 'conv1/BatchNorm/moving_variance'])

    # block1
    neural_net_pb.convert_start_block(pytorch_resnet.block1.unit_1, 'block1', 'unit_1')
    neural_net_pb.convert_mid_block(pytorch_resnet.block1.unit_2, 'block1', 'unit_2')
    neural_net_pb.convert_end_block(pytorch_resnet.block1.unit_3, 'block1', 'unit_3')

    # block2
    neural_net_pb.convert_start_block(pytorch_resnet.block2.unit_1, 'block2', 'unit_1')
    neural_net_pb.convert_mid_block(pytorch_resnet.block2.unit_2, 'block2', 'unit_2')
    neural_net_pb.convert_mid_block(pytorch_resnet.block2.unit_3, 'block2', 'unit_3')
    neural_net_pb.convert_end_block(pytorch_resnet.block2.unit_4, 'block2', 'unit_4')

    # block3
    neural_net_pb.convert_start_block(pytorch_resnet.block3.unit_1, 'block3', 'unit_1')
    neural_net_pb.convert_mid_block(pytorch_resnet.block3.unit_2, 'block3', 'unit_2')
    neural_net_pb.convert_mid_block(pytorch_resnet.block3.unit_3, 'block3', 'unit_3')
    neural_net_pb.convert_mid_block(pytorch_resnet.block3.unit_4, 'block3', 'unit_4')
    neural_net_pb.convert_mid_block(pytorch_resnet.block3.unit_5, 'block3', 'unit_5')
    neural_net_pb.convert_end_block(pytorch_resnet.block3.unit_6, 'block3', 'unit_6')

    # block4
    neural_net_pb.convert_start_block(pytorch_resnet.block4.unit_1, 'block4', 'unit_1')
    neural_net_pb.convert_mid_block(pytorch_resnet.block4.unit_2, 'block4', 'unit_2')
    neural_net_pb.convert_mid_block(pytorch_resnet.block4.unit_3, 'block4', 'unit_3')

    pytorch_resnet.id.tf_fc.weight[:, :, :, :] = neural_net_pb.convert_to_torch(neural_net_pb.weights['fc-id/weights'], permute=True)
    pytorch_resnet.id.add_bais[:] = neural_net_pb.convert_to_torch(neural_net_pb.weights['fc-id/biases'])
    pytorch_resnet.ex.tf_fc.weight[:, :, :, :] = neural_net_pb.convert_to_torch(neural_net_pb.weights['fc-ex/weights'], permute=True)
    pytorch_resnet.ex.add_bais[:] = neural_net_pb.convert_to_torch(neural_net_pb.weights['fc-ex/biases'])
    pytorch_resnet.tex.tf_fc.weight[:, :, :, :] = neural_net_pb.convert_to_torch(neural_net_pb.weights['fc-tex/weights'], permute=True)
    pytorch_resnet.tex.add_bais[:] = neural_net_pb.convert_to_torch(neural_net_pb.weights['fc-tex/biases'])
    pytorch_resnet.angles.tf_fc.weight[:, :, :, :] = neural_net_pb.convert_to_torch(neural_net_pb.weights['fc-angles/weights'], permute=True)
    pytorch_resnet.angles.add_bais[:] = neural_net_pb.convert_to_torch(neural_net_pb.weights['fc-angles/biases'])
    pytorch_resnet.gamma.tf_fc.weight[:, :, :, :] = neural_net_pb.convert_to_torch(neural_net_pb.weights['fc-gamma/weights'], permute=True)
    pytorch_resnet.gamma.add_bais[:] = neural_net_pb.convert_to_torch(neural_net_pb.weights['fc-gamma/biases'])
    pytorch_resnet.xy.tf_fc.weight[:, :, :, :] = neural_net_pb.convert_to_torch(neural_net_pb.weights['fc-XY/weights'], permute=True)
    pytorch_resnet.xy.add_bais[:] = neural_net_pb.convert_to_torch(neural_net_pb.weights['fc-XY/biases'])
    pytorch_resnet.z.tf_fc.weight[:, :, :, :] = neural_net_pb.convert_to_torch(neural_net_pb.weights['fc-Z/weights'], permute=True)
    pytorch_resnet.z.add_bais[:] = neural_net_pb.convert_to_torch(neural_net_pb.weights['fc-Z/biases'])

    pytorch_resnet.eval()
    path = 'path to temp_pytorch_converted_model.pt'  # TODO: path to pytorch model save
    torch.save(pytorch_resnet.state_dict(), path)
    pytorch_resnet_loaded = resnet50()
    pytorch_resnet_loaded.load_state_dict(torch.load(path))
    pytorch_resnet_loaded.eval()
    neural_net_pb.sanity_check(pytorch_resnet_loaded)
    # print(pytorch_resnet)

