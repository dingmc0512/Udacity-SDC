import tensorflow as tf
from settings import *
from layers import *

IS_TRAIN = None
NUM_FEAT = 3

def get_conv_weight(name,kshape,wd=0.0005):
	with tf.device('/cpu:0'):
		var=tf.get_variable(name,shape=kshape,initializer=tf.contrib.layers.xavier_initializer())
	if wd!=0:
		weight_decay = tf.nn.l2_loss(var)*wd
		tf.add_to_collection('weightdecay_losses', weight_decay)
	return var

def grouped_conv3d(name,l_input,in_channels,out_channels,num_groups,strides=[1,1,1,1,1]):
	with tf.variable_scope(name) as scope:
		sz = in_channels // num_groups
		conv_side_layers = [
			tf.nn.conv3d(l_input[..., i*sz : i*sz+sz], get_conv_weight(name=name+"_"+str(i), 
													kshape=[1,1,1, sz, out_channels//num_groups]),
													strides=strides,padding='SAME') for i in
			range(num_groups)]
		out = tf.concat(conv_side_layers, axis=-1)
		out = tf.layers.batch_normalization(out,training=IS_TRAIN,name=name+'_bn')
		out = tf.nn.relu(out)
	return out

def channel_shuffle(name, l_input, num_groups):
    with tf.variable_scope(name) as scope:
        n, d, h, w, c = l_input.shape.as_list()
        out = tf.reshape(l_input, [-1, d, h, w, num_groups, c // num_groups])
        out = tf.transpose(out, [0, 1, 2, 3, 5, 4])
        out = tf.reshape(out, [-1, d, h, w, c])
    return out

def convS(name,l_input,in_channels,out_channels):
	return tf.nn.bias_add(tf.nn.conv3d(l_input,get_conv_weight(name=name,
															   kshape=[1,3,3,in_channels,out_channels]),
															   strides=[1,1,1,1,1],padding='SAME'),
											  get_conv_weight(name+'_bias',[out_channels],0))
def convT(name,l_input,in_channels,out_channels):
	return tf.nn.bias_add(tf.nn.conv3d(l_input,get_conv_weight(name=name,
															   kshape=[1,1,1,in_channels,out_channels]),
															   strides=[1,1,1,1,1],padding='SAME'),
											  get_conv_weight(name+'_bias',[out_channels],0))

#build the bottleneck struction of each block.
class Bottleneck():
	def __init__(self,l_input,inplanes,planes,stride=[1,1,1,1,1],n_s=0):
		
		self.X_input=l_input
		self.planes=planes
		self.inplanes=inplanes
		self.ST_struc=('A','B','C')
		self.len_ST=len(self.ST_struc)
		self.id=n_s
		self.ST=list(self.ST_struc)[self.id % self.len_ST]
		self.stride=stride

	#P3D has three types of bottleneck sub-structions.
	def ST_A(self,name,x):
		x=convS(name+'_S',x,self.planes,self.planes)
		x=tf.layers.batch_normalization(x,training=IS_TRAIN,name=name+'_S_bn')
		x=tf.nn.relu(x)
		x=convT(name+'_T',x,self.planes,self.planes)
		x=tf.layers.batch_normalization(x,training=IS_TRAIN,name=name+'_T_bn')
		x=tf.nn.relu(x)
		return x
	
	def ST_B(self,name,x):
		tmp_x=convS(name+'_S',x,self.planes,self.planes)
		tmp_x=tf.layers.batch_normalization(tmp_x,training=IS_TRAIN,name=name+'_S_bn')
		tmp_x=tf.nn.relu(tmp_x)
		x=convT(name+'_T',x,self.planes,self.planes)
		x=tf.layers.batch_normalization(x,training=IS_TRAIN,name=name+'_T_bn')
		x=tf.nn.relu(x)
		return x+tmp_x
	
	def ST_C(self,name,x):
		x=convS(name+'_S',x,self.planes,self.planes)
		x=tf.layers.batch_normalization(x,training=IS_TRAIN,name=name+'_S_bn')
		x=tf.nn.relu(x)
		tmp_x=convT(name+'_T',x,self.planes,self.planes)
		tmp_x=tf.layers.batch_normalization(tmp_x,training=IS_TRAIN,name=name+'_T_bn')
		tmp_x=tf.nn.relu(tmp_x)
		return x+tmp_x
	
	def infer(self):
		residual=self.X_input
		
		out = grouped_conv3d('group_conv3_{}_1'.format(self.id), self.X_input, self.inplanes, self.planes, NUM_GROUPS, self.stride)
		out = channel_shuffle('channel_shuffle_{}'.format(self.id), out, NUM_GROUPS)
		out = tf.nn.relu(out)

		if self.ST=='A':
			out=self.ST_A('STA_{}_2'.format(self.id),out)
		elif self.ST=='B':
			out=self.ST_B('STB_{}_2'.format(self.id),out)
		elif self.ST=='C':
			out=self.ST_C('STC_{}_2'.format(self.id),out)
				
		out = grouped_conv3d('group_conv3_{}_3'.format(self.id),out, self.planes, self.planes*BLOCK_EXPANSION, NUM_GROUPS)
		   
		residual=tf.nn.conv3d(residual,get_conv_weight('dw3d_{}'.format(self.id),[1,1,1,self.inplanes,self.planes*BLOCK_EXPANSION]),
							  strides=self.stride,padding='SAME')
		residual=tf.layers.batch_normalization(residual,training=IS_TRAIN,name='dw3d_{}_bn'.format(self.id))
		
		out+=residual
		out=tf.nn.relu(out)
		
		return out

# build a singe block of p3d
class make_block():
	def __init__(self,_X,planes,num,inplanes,cnt,stride=1):
		self.input=_X
		self.planes=planes
		self.inplanes=inplanes
		self.num=num
		self.cnt=cnt
		if stride!=1:
			self.downsample=[1,1,2,2,1]
		else:
			self.downsample=[1,1,1,1,1]

	def infer(self):
		x=Bottleneck(self.input,self.inplanes,self.planes,self.downsample,n_s=self.cnt).infer()
		self.cnt+=1
		self.inplanes=BLOCK_EXPANSION*self.planes
		for i in range(1,self.num):
			x=Bottleneck(x,self.inplanes,self.planes,n_s=self.cnt).infer()
			self.cnt+=1
		return x

#build structure of the p3d network.
def inference(frames,feature_size,_dropout):
	global IS_TRAIN
	IS_TRAIN = True
	len_depth = 1
	list_depth = [1]*(3-len_depth) + [2]*len_depth

	cnt=0
	#print('input_shape:', frames.shape.as_list())
	conv1_custom=tf.nn.conv3d(frames,get_conv_weight('firstconv1',[1,7,7,RGB_CHANNEL,64]),strides=[1,1,2,2,1],padding='SAME')
	conv1_custom_bn=tf.layers.batch_normalization(conv1_custom,training=IS_TRAIN,name='firstconv1')
	conv1_custom_bn_relu=tf.nn.relu(conv1_custom_bn)
	#print('conv1_shape:', conv1_custom.shape.as_list())

	x=tf.nn.max_pool3d(conv1_custom_bn_relu,[1,1,3,3,1],strides=[1,1,2,2,1],padding='SAME')
	#print('max3d_shape:', x.shape.as_list())
	b1=make_block(x,64,3,64,cnt)
	x=b1.infer()
	#print('block_shape:', x.shape.as_list())
	cnt=b1.cnt
   
	x=tf.nn.max_pool3d(x,[1,list_depth[0],1,1,1],strides=[1,list_depth[0],1,1,1],padding='SAME')
	#print('max3d_shape:', x.shape.as_list())
	b2=make_block(x,128,4,256,cnt,stride=2)
	x=b2.infer()
	#print('block_shape:', x.shape.as_list())
	cnt=b2.cnt
	x=tf.nn.max_pool3d(x,[1,list_depth[1],1,1,1],strides=[1,list_depth[1],1,1,1],padding='SAME')
	
	#print('max3d_shape:', x.shape.as_list())
	b3=make_block(x,256,6,512,cnt,stride=2)
	x=b3.infer()
	#print('block_shape:', x.shape.as_list())
	cnt=b3.cnt
	x=tf.nn.max_pool3d(x,[1,list_depth[2],1,1,1],strides=[1,list_depth[2],1,1,1],padding='SAME')
	
	#print('max3d_shape:', x.shape.as_list())
	x=make_block(x,512,3,1024,cnt,stride=2).infer()

	#print('block_shape:', x.shape.as_list())
	#Caution:make sure avgpool on the input which has the same shape as kernelsize has been setted padding='VALID'
	x=tf.nn.avg_pool3d(x,[1,1,5,5,1],strides=[1,1,1,1,1],padding='VALID')
	#print('avg2d_shape:', x.shape.as_list())

	x=tf.reshape(x,shape=[-1,2048])
	#print('fc_re_shape:', x.shape.as_list())
 
	x=tf.nn.dropout(x,keep_prob=_dropout)
	x=tf.layers.dense(x,feature_size,name='fc')
	x=tf.reshape(x,shape=[-1,NUM_FEAT,feature_size])
	#print('fc_re_shape:', x.shape.as_list())

	return x
