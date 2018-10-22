import tensorflow as tf
from settings import *
from layers import *

IS_TRAIN = None


def get_conv_weight(name,kshape,wd=0.0005):
    with tf.device('/cpu:0'):
        var=tf.get_variable(name,shape=kshape,initializer=tf.contrib.layers.xavier_initializer())
    if wd!=0:
        weight_decay = tf.nn.l2_loss(var)*wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var

def SE_2D(self, input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name):
        squeeze = tf.reduce_mean(input_x, [1,2])
        excitation = tf.layers.dense(squeeze, units=out_dim / ratio, name=layer_name+'_fc1')
        excitation = tf.nn.relu(excitation)
        excitation = tf.layers.dense(excitation, units=out_dim, name=layer_name+'_fc2')
        excitation = tf.nn.sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        scale = input_x * excitation
    return scale

def SE_3D(self, input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name):
        squeeze = tf.reduce_mean(input_x, [1,2,3])
        excitation = tf.layers.dense(squeeze, units=out_dim / ratio, name=layer_name+'_fc1')
        excitation = tf.nn.relu(excitation)
        excitation = tf.layers.dense(excitation, units=out_dim, name=layer_name+'_fc2')
        excitation = tf.nn.sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1,1,1,1,out_dim])
        scale = input_x * excitation
    return scale

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

def __tf_scan_depthwise_conv3d(x, w, stride, padding='SAME'):
    def conv3d(_, _input):
        _x = _input[0]
        _w = _input[1]
        _out = tf.nn.conv3d(_x,_w,stride,padding)
        return _out

    with tf.variable_scope('tf_scan_depth_conv3d') as scope:
        kernel_scan = tf.transpose(w,[3,0,1,2,4])
        kernel_scan = tf.expand_dims(kernel_scan,axis=4)
        x_scan = tf.transpose(x,[4,0,1,2,3])
        x_scan = tf.expand_dims(x_scan,axis=-1)
        xshape = x.shape.as_list()
        xshape[-1] = 1
        
        out = tf.scan(conv3d, (x_scan,kernel_scan), initializer = tf.zeros((xshape)))
        out = tf.transpose(out,[1,2,3,4,0,5])
        out = tf.reshape(out,x.shape)
    return out


def __py_split_depthwise_conv3d(x, w, stride, padding='SAME'):
    with tf.variable_scope('tf_split_depth_conv3d'):
        depth = x.get_shape()[-1].value
        conv_depth_layers = [
            tf.nn.conv3d(x[..., i:i+1], w[..., i:i+1, :], stride, padding) for i in
            range(depth)]
        out = tf.concat(conv_depth_layers, axis=-1)
    return out


def __tf_split_depthwise_conv3d(x, w, stride, padding='SAME'):
    with tf.variable_scope('tf_split_depth_conv3d'):
        inputs_splits = tf.split(x,x.shape[-1],-1)
        print(inputs_splits[0].shape.as_list())
        kernel_splits = tf.split(w,w.shape[-2],-2)
        print(kernel_splits[0].shape.as_list())
        conv_depth_layers = [
            tf.nn.conv3d(inputs, kernel, stride, padding) 
            for inputs, kernel in zip(inputs_splits, kernel_splits)]
        out = tf.concat(conv_depth_layers, axis=-1)
    return out
    

def __depthwise_conv3d_p(name, x, w=None, kernel_size=(3, 3, 3), padding='SAME', stride=(1, 1, 1),
                         initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], stride[2], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], kernel_size[2], x.shape[-1], 1]

        with tf.name_scope('layer_weights'):
            if w is None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            __variable_summaries(w)
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [x.shape[-1]], initializer=tf.constant_initializer(bias))
            __variable_summaries(bias)
        with tf.name_scope('layer_conv3d'):
            conv = __tf_scan_depthwise_conv3d(x, w, stride, padding)
            out = tf.nn.bias_add(conv, bias)
    return out

def convS(name,l_input,in_channels,out_channels):
    return tf.nn.bias_add(tf.nn.conv3d(l_input,get_conv_weight(name=name,
                                                               kshape=[1,3,3,in_channels,out_channels]),
                                                               strides=[1,1,1,1,1],padding='SAME'),
                                              get_conv_weight(name+'_bias',[out_channels],0))
def convT(name,l_input,in_channels,out_channels):
    return tf.nn.bias_add(tf.nn.conv3d(l_input,get_conv_weight(name=name,
                                                               kshape=[3,1,1,in_channels,out_channels]),
                                                               strides=[1,1,1,1,1],padding='SAME'),
                                              get_conv_weight(name+'_bias',[out_channels],0))

'''
def convS(name,l_input,in_channels,out_channels):
    return tf.nn.bias_add(__tf_scan_depthwise_conv3d(l_input,get_conv_weight(name=name,
                                                               kshape=[1,3,3,in_channels,1]),
                                                               stride=[1,1,1,1,1],padding='SAME'),
                                              get_conv_weight(name+'_bias',[in_channels],0))
def convT(name,l_input,in_channels,out_channels):
    return tf.nn.bias_add(__tf_scan_depthwise_conv3d(l_input,get_conv_weight(name=name,
                                                               kshape=[3,1,1,in_channels,1]),
                                                               stride=[1,1,1,1,1],padding='SAME'),
                                              get_conv_weight(name+'_bias',[in_channels],0))
'''
'''
def convS(name,l_input,in_channels,out_channels):
    shape = l_input.shape.as_list()
    dw_channels = shape[1] * shape[-1]
    out = tf.transpose(l_input, [0,2,3,1,4])
    out = tf.reshape(out, (shape[0], shape[2], shape[3], dw_channels))
    out = tf.nn.bias_add(tf.nn.depthwise_conv2d(out, get_conv_weight(name=name+'_dw_conv',
                                                                kshape=[3,3,dw_channels,1]),
                                                                strides=[1,1,1,1],padding='SAME'),
                                                get_conv_weight(name+'_dw_bias',[dw_channels],0))

    out = tf.reshape(out, (shape[0], shape[2], shape[3], shape[1], shape[4]))
    out = tf.transpose(out, [0,3,1,2,4])
    out = tf.nn.bias_add(tf.nn.conv3d(out,get_conv_weight(name=name+'_n_conv',
                                                                kshape=[1,1,1,in_channels,out_channels]),
                                                                strides=[1,1,1,1,1],padding='SAME'),
                                                get_conv_weight(name+'_n_bias',[out_channels],0))
    return out

def convT(name,l_input,in_channels,out_channels):
    out = tf.nn.bias_add(__py_split_depthwise_conv3d(l_input,get_conv_weight(name=name,
                                                               kshape=[3,1,1,in_channels,1]),
                                                               stride=[1,1,1,1,1],padding='SAME'),
                                              get_conv_weight(name+'_bias',[in_channels],0))

    out = tf.nn.bias_add(tf.nn.conv3d(out,get_conv_weight(name=name+'_n_conv',
                                                                kshape=[1,1,1,in_channels,out_channels]),
                                                                strides=[1,1,1,1,1],padding='SAME'),
                                                get_conv_weight(name+'_n_bias',[out_channels],0))
    return out
'''

#build the bottleneck struction of each block.
class Bottleneck():
    def __init__(self,l_input,inplanes,planes,stride=1,downsample='',n_s=0,depth_3d=47):
        
        self.X_input=l_input
        self.downsample=downsample
        self.planes=planes
        self.inplanes=inplanes
        self.depth_3d=depth_3d
        self.ST_struc=('A','B','C')
        self.len_ST=len(self.ST_struc)
        self.id=n_s
        self.n_s=n_s
        self.ST=list(self.ST_struc)[self.id % self.len_ST]
        self.stride_p=[1,1,1,1,1]
       
        if self.downsample!='':
            self.stride_p=[1,1,2,2,1]
        if n_s<self.depth_3d:
            if n_s==0:
                self.stride_p=[1,1,1,1,1]
        else:
            if n_s==self.depth_3d:
                self.stride_p=[1,2,2,2,1]
            else:
                self.stride_p=[1,1,1,1,1]
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

    def SE_2D(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):
            squeeze = tf.reduce_mean(input_x, [1,2])
            excitation = tf.layers.dense(squeeze, units=out_dim / ratio, name=layer_name+'_fc1')
            excitation = tf.nn.relu(excitation)
            excitation = tf.layers.dense(excitation, units=out_dim, name=layer_name+'_fc2')
            excitation = tf.nn.sigmoid(excitation)
            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            scale = input_x * excitation
        return scale

    def SE_3D(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):
            squeeze = tf.reduce_mean(input_x, [1,2,3])
            excitation = tf.layers.dense(squeeze, units=out_dim / ratio, name=layer_name+'_fc1')
            excitation = tf.nn.relu(excitation)
            excitation = tf.layers.dense(excitation, units=out_dim, name=layer_name+'_fc2')
            excitation = tf.nn.sigmoid(excitation)
            excitation = tf.reshape(excitation, [-1,1,1,1,out_dim])
            scale = input_x * excitation
        return scale
    
    def infer(self):
        residual=self.X_input
        if self.n_s<self.depth_3d:
            
            out = grouped_conv3d('group_conv3_{}_1'.format(self.id), self.X_input, self.inplanes, self.planes, NUM_GROUPS, self.stride_p)
            out = channel_shuffle('channel_shuffle_{}'.format(self.id), out, NUM_GROUPS)
            '''
            out=tf.nn.conv3d(self.X_input,get_conv_weight('conv3_{}_1'.format(self.id),[1,1,1,self.inplanes,self.planes]),
                             strides=self.stride_p,padding='SAME')
            out=tf.layers.batch_normalization(out,training=IS_TRAIN,name='conv3_{}_1_bn'.format(self.id))
            '''
        else:
            param=self.stride_p
            param.pop(1)
            out=tf.nn.conv2d(self.X_input,get_conv_weight('conv2_{}_1'.format(self.id),[1,1,self.inplanes,self.planes]),
                             strides=param,padding='SAME')
            out=tf.layers.batch_normalization(out,training=IS_TRAIN,name='conv2_{}_1_bn'.format(self.id))
    
        out=tf.nn.relu(out)    
        if self.id<self.depth_3d:
            if self.ST=='A':
                out=self.ST_A('STA_{}_2'.format(self.id),out)
            elif self.ST=='B':
                out=self.ST_B('STB_{}_2'.format(self.id),out)
            elif self.ST=='C':
                out=self.ST_C('STC_{}_2'.format(self.id),out)
        else:
            out=tf.nn.conv2d(out,get_conv_weight('conv2_{}_2'.format(self.id),[3,3,self.planes,self.planes]),
                                  strides=[1,1,1,1],padding='SAME')
            out=tf.layers.batch_normalization(out,training=IS_TRAIN,name='conv2_{}_2_bn'.format(self.id))
            out=tf.nn.relu(out)

        if self.n_s<self.depth_3d:
            
            out = grouped_conv3d('group_conv3_{}_3'.format(self.id),out, self.planes, self.planes*BLOCK_EXPANSION, NUM_GROUPS)
            '''
            out=tf.nn.conv3d(out,get_conv_weight('conv3_{}_3'.format(self.id),[1,1,1,self.planes,self.planes*BLOCK_EXPANSION]),
                             strides=[1,1,1,1,1],padding='SAME')
            out=tf.layers.batch_normalization(out,training=IS_TRAIN,name='conv3_{}_3_bn'.format(self.id))
            '''
            #out=self.SE_3D(out, self.planes*BLOCK_EXPANSION, 2, 'SE_3D_{}'.format(self.id))
            
        else:
            out=tf.nn.conv2d(out,get_conv_weight('conv2_{}_3'.format(self.id),[1,1,self.planes,self.planes*BLOCK_EXPANSION]),
                             strides=[1,1,1,1],padding='SAME')
            out=tf.layers.batch_normalization(out,training=IS_TRAIN,name='conv2_{}_3_bn'.format(self.id))
            #out=self.SE_2D(out, self.planes*BLOCK_EXPANSION, 2, 'SE_2D_{}'.format(self.id))
           
        if len(self.downsample)==1:
            residual=tf.nn.conv2d(residual,get_conv_weight('dw2d_{}'.format(self.id),[1,1,self.inplanes,self.planes*BLOCK_EXPANSION]),
                                  strides=[1,2,2,1],padding='SAME')
            residual=tf.layers.batch_normalization(residual,training=IS_TRAIN,name='dw2d_{}_bn'.format(self.id))
        elif len(self.downsample)==2:
            residual=tf.nn.conv3d(residual,get_conv_weight('dw3d_{}'.format(self.id),[1,1,1,self.inplanes,self.planes*BLOCK_EXPANSION]),
                                  strides=self.downsample[1],padding='SAME')
            residual=tf.layers.batch_normalization(residual,training=IS_TRAIN,name='dw3d_{}_bn'.format(self.id))
        
        out+=residual
        out=tf.nn.relu(out)
        
        return out

# build a singe block of p3d,depth_3d=47 means p3d-152(199)
# build a singe block of p3d,depth_3d=13 means p3d-50
class make_block():
    def __init__(self,_X,planes,num,inplanes,cnt,depth_3d=13,stride=1):
        self.input=_X
        self.planes=planes
        self.inplanes=inplanes
        self.num=num
        self.cnt=cnt
        self.depth_3d=depth_3d
        self.stride=stride
        if self.cnt<depth_3d:
            if self.cnt==0:
                stride_p=[1,1,1,1,1]
            else:
                stride_p=[1,1,2,2,1]
            if stride!=1 or inplanes!=planes*BLOCK_EXPANSION:
                self.downsample=['3d',stride_p]
        else:
            if stride!=1 or inplanes!=planes*BLOCK_EXPANSION:
                self.downsample=['2d']
    def infer(self):
        x=Bottleneck(self.input,self.inplanes,self.planes,self.stride,self.downsample,n_s=self.cnt,depth_3d=self.depth_3d).infer()
        self.cnt+=1
        self.inplanes=BLOCK_EXPANSION*self.planes
        for i in range(1,self.num):
            x=Bottleneck(x,self.inplanes,self.planes,n_s=self.cnt,depth_3d=self.depth_3d).infer()
            self.cnt+=1
        return x

#build structure of the p3d network.
def inference_p3d(frames,feature_size,_dropout):
    global IS_TRAIN
    IS_TRAIN = True

    cnt=0
    print('input_shape:', frames.shape.as_list())
    conv1_custom=tf.nn.conv3d(frames,get_conv_weight('firstconv1',[1,7,7,RGB_CHANNEL,64]),strides=[1,1,2,2,1],padding='SAME')
    conv1_custom_bn=tf.layers.batch_normalization(conv1_custom,training=IS_TRAIN,name='firstconv1')
    conv1_custom_bn_relu=tf.nn.relu(conv1_custom_bn)
    print('conv1_shape:', conv1_custom.shape.as_list())

    x=tf.nn.max_pool3d(conv1_custom_bn_relu,[1,2,3,3,1],strides=[1,2,2,2,1],padding='SAME')
    print('max3d_shape:', x.shape.as_list())
    b1=make_block(x,64,3,64,cnt)
    x=b1.infer()
    print('block_shape:', x.shape.as_list())
    cnt=b1.cnt
   
    x=tf.nn.max_pool3d(x,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    print('max3d_shape:', x.shape.as_list())
    b2=make_block(x,128,4,256,cnt,stride=2)
    x=b2.infer()
    print('block_shape:', x.shape.as_list())
    cnt=b2.cnt
    x=tf.nn.max_pool3d(x,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    
    print('max3d_shape:', x.shape.as_list())
    b3=make_block(x,256,6,512,cnt,stride=2)
    x=b3.infer()
    print('block_shape:', x.shape.as_list())
    cnt=b3.cnt
    x=tf.nn.max_pool3d(x,[1,2,1,1,1],strides=[1,2,1,1,1],padding='SAME')
    
    print('max3d_shape:', x.shape.as_list())
    shape=x.shape.as_list()
    x=tf.reshape(x,shape=[-1,shape[2],shape[3],shape[4]])
    print('2d_re_shape:', x.shape.as_list())
    x=make_block(x,512,3,1024,cnt,stride=2).infer()

    print('block_shape:', x.shape.as_list())
	#Caution:make sure avgpool on the input which has the same shape as kernelsize has been setted padding='VALID'
    x=tf.nn.avg_pool(x,[1,5,5,1],strides=[1,1,1,1],padding='VALID')
    print('avg2d_shape:', x.shape.as_list())

    x=tf.reshape(x,shape=[-1,2048])
    print('fc_re_shape:', x.shape.as_list())
 
    x=tf.nn.dropout(x,keep_prob=_dropout)
    x=tf.layers.dense(x,feature_size,name='fc')

    return x
