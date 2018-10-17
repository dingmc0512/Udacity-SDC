import tensorflow as tf
import tensorflow.contrib.slim as slim

IS_TRAIN = None

#Define base convolution layers for 'pseudo-3d convolution'(P3D):
def convS(_X,out_channels,kernel_size=[1,3,3],stride=1,padding='VALID'):
    return slim.conv3d(_X,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                   biases_initializer=None)

def convT(_X,out_channels,kernel_size=[3,1,1],stride=1,padding='VALID'):
    return slim.conv3d(_X,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,
                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       biases_initializer=None)

def ST_A(_X,out_channels,s_kernel,t_kernel,stride,padding):
	x=convS(_X,out_channels,s_kernel,stride,padding)
	x=tf.layers.batch_normalization(x,training=IS_TRAIN)
	x=tf.nn.relu(x)
	x=convT(x,out_channels,t_kernel,stride,padding)
	x=tf.layers.batch_normalization(x,training=IS_TRAIN)
	x=tf.nn.relu(x)
	return x
    
def ST_B(_X,out_channels,s_kernel,t_kernel,stride,padding):
    tmp_x=convS(_X,out_channels,s_kernel,stride,padding)
    tmp_x=tf.layers.batch_normalization(tmp_x,training=IS_TRAIN)
    tmp_x=tf.nn.relu(tmp_x)
    
    x=convT(_X,out_channels,t_kernel,stride,padding)
    x=tf.layers.batch_normalization(x,training=IS_TRAIN)
    x=tf.nn.relu(x)
    return x+tmp_x
    
def ST_C(_X,out_channels,s_kernel,t_kernel,stride,padding):
    x=convS(_X,out_channels,s_kernel,stride,padding)
    x=tf.layers.batch_normalization(x,training=IS_TRAIN)
    x=tf.nn.relu(x)
   
    tmp_x=convT(x,out_channels,t_kernel,stride,padding)
    tmp_x=tf.layers.batch_normalization(tmp_x,training=IS_TRAIN)
    tmp_x=tf.nn.relu(tmp_x)
    return x+tmp_x
    
def p_conv3d(_X,out_channels,kernel_size=[3,3,3],stride=1,padding='SAME',tp='B',is_train=True):
	IS_TRAIN = is_train

	s_kernel=kernel_size[1:3]
	s_kernel.insert(0,1)
	t_kernel=[1,1]
	t_kernel.insert(0,kernel_size[0])
	if tp=='B':
		return ST_B(_X,out_channels,s_kernel,t_kernel,stride,padding)
	else:
		return ST_C(_X,out_channels,s_kernel,t_kernel,stride,padding)
