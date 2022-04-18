import tensorflow as tf


def load_graph(graph_filename):
	with tf.gfile.GFile(graph_filename,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	return graph_def


if __name__ == '__main__':
	graph_def = load_graph('path to FaceReconModel.pb')  # TODO: path to FaceReconModel.pb
	print(graph_def)