import tensorflow as tf
from settings import *
slim = tf.contrib.slim


layer_norm = lambda x: tf.contrib.layers.layer_norm(inputs=x, center=True, scale=True, activation_fn=None, trainable=True)

def inference(inputs, keep_prob, seq_len):
	#print('input_shape:', inputs.shape.as_list())
	net = slim.convolution(inputs, num_outputs=64, kernel_size=[3,12,12], stride=[1,6,6], padding="VALID")
	net = tf.nn.dropout(x=net, keep_prob=keep_prob)
	aux1 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [BATCH_SIZE, seq_len, -1]), 128, activation_fn=None)
	#print('conv1_shape:', net.shape.as_list())

	net = slim.convolution(net, num_outputs=64, kernel_size=[2,5,5], stride=[1,2,2], padding="VALID")
	net = tf.nn.dropout(x=net, keep_prob=keep_prob)
	aux2 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [BATCH_SIZE, seq_len, -1]), 128, activation_fn=None)
	#print('conv2_shape:', net.shape.as_list())

	net = slim.convolution(net, num_outputs=64, kernel_size=[2,5,5], stride=[1,1,1], padding="VALID")
	net = tf.nn.dropout(x=net, keep_prob=keep_prob)
	aux3 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [BATCH_SIZE, seq_len, -1]), 128, activation_fn=None)
	#print('conv3_shape:', net.shape.as_list())

	net = slim.convolution(net, num_outputs=64, kernel_size=[2,5,5], stride=[1,1,1], padding="VALID")
	net = tf.nn.dropout(x=net, keep_prob=keep_prob)
	# at this point the tensor 'net' is of shape BATCH_SIZE x seq_len x ...
	aux4 = slim.fully_connected(tf.reshape(net, [BATCH_SIZE, seq_len, -1]), 128, activation_fn=None)
	#print('conv4_shape:', net.shape.as_list())

	net = slim.fully_connected(tf.reshape(net, [BATCH_SIZE, seq_len, -1]), 1024, activation_fn=tf.nn.relu)
	#print('fc1_shape:', net.shape.as_list())
	net = tf.nn.dropout(x=net, keep_prob=keep_prob)
	net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
	#print('fc2_shape:', net.shape.as_list())
	net = tf.nn.dropout(x=net, keep_prob=keep_prob)
	net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu)
	#print('fc3_shape:', net.shape.as_list())
	net = tf.nn.dropout(x=net, keep_prob=keep_prob)
	net = slim.fully_connected(net, 128, activation_fn=None)
	#print('fc4_shape:', net.shape.as_list())
	return layer_norm(tf.nn.elu(net + aux1 + aux2 + aux3 + aux4)) # aux[1-4] are residual connections (shortcuts)