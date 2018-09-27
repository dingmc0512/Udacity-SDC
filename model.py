import tensorflow as tf
import P3D
import T3D
from SamplingRNNCell import SamplingRNNCell
from settings import *
slim = tf.contrib.slim

layer_norm = lambda x: tf.contrib.layers.layer_norm(inputs=x, center=True, scale=True, activation_fn=None, trainable=True)

def ST_Conv_Vision_Simple(image, keep_prob, batch_size, seq_len, scope=None, reuse=None):
    video = tf.reshape(image, shape=[batch_size, LEFT_CONTEXT + seq_len, HEIGHT, WIDTH, CHANNELS])
    with tf.variable_scope(scope, 'Vision', [image], reuse=reuse):
        print('input_shape:', video.shape.as_list())
        net = slim.convolution(video, num_outputs=64, kernel_size=[3,12,12], stride=[1,6,6], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        aux1 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128, activation_fn=None)
        print('conv1_shape:', net.shape.as_list())

        net = slim.convolution(net, num_outputs=64, kernel_size=[2,5,5], stride=[1,2,2], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        aux2 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128, activation_fn=None)
        print('conv2_shape:', net.shape.as_list())

        net = slim.convolution(net, num_outputs=64, kernel_size=[2,5,5], stride=[1,1,1], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        aux3 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128, activation_fn=None)
        print('conv3_shape:', net.shape.as_list())

        net = slim.convolution(net, num_outputs=64, kernel_size=[2,5,5], stride=[1,1,1], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        # at this point the tensor 'net' is of shape batch_size x seq_len x ...
        aux4 = slim.fully_connected(tf.reshape(net, [batch_size, seq_len, -1]), 128, activation_fn=None)
        print('conv4_shape:', net.shape.as_list())

        net = slim.fully_connected(tf.reshape(net, [batch_size, seq_len, -1]), 1024, activation_fn=tf.nn.relu)
        print('fc1_shape:', net.shape.as_list())
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
        print('fc2_shape:', net.shape.as_list())
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu)
        print('fc3_shape:', net.shape.as_list())
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        net = slim.fully_connected(net, 128, activation_fn=None)
        print('fc4_shape:', net.shape.as_list())
        return layer_norm(tf.nn.elu(net + aux1 + aux2 + aux3 + aux4)) # aux[1-4] are residual connections (shortcuts)


def CNN_Vision_Simple(image, keep_prob, batch_size, seq_len, scope=None, reuse=None):
    video = tf.reshape(image, shape=[batch_size, LEFT_CONTEXT + seq_len, CROP_SIZE, CROP_SIZE, CHANNELS])
    with tf.variable_scope(scope, 'Vision', [image], reuse=reuse):
        seq_net = []
        for idx in range(seq_len):
            net = P3D.inference_p3d(video[:,LEFT_CONTEXT+idx,:,:,:],VISION_FEATURE_SIZE,keep_prob,batch_size,(keep_prob!=1.0))
            seq_net.append(net)
        return tf.stack(seq_net, axis=1)


def P3D_Vision_Simple(image, keep_prob, batch_size, seq_len, scope=None, reuse=None):
    video = tf.reshape(image, shape=[batch_size, LEFT_CONTEXT + seq_len, CROP_SIZE, CROP_SIZE, CHANNELS])
    with tf.variable_scope(scope, 'Vision', [image], reuse=reuse):
        seq_net = []
        for idx in range(seq_len):
            net = P3D.inference_p3d(video[:,idx+1:LEFT_CONTEXT+idx+1,:,:,:],VISION_FEATURE_SIZE,keep_prob,batch_size,(keep_prob!=1.0))
            seq_net.append(net)
        return tf.stack(seq_net, axis=1)


def T3D_Vision_Simple(image, keep_prob, batch_size, seq_len, scope=None, reuse=None):
    video = tf.reshape(image, shape=[batch_size, LEFT_CONTEXT + seq_len, CROP_SIZE, CROP_SIZE, CHANNELS])
    with tf.variable_scope(scope, 'Vision', [image], reuse=reuse):
        seq_net = []
        for idx in range(seq_len):
            net = T3D.inference_t3d(video[:,idx+1:LEFT_CONTEXT+idx+1,:,:,:],VISION_FEATURE_SIZE,keep_prob,batch_size,(keep_prob!=1.0))
            seq_net.append(net)
        return tf.stack(seq_net, axis=1)


def inference(input_images, targets_normalized, keep_prob):
    input_images = -1.0 + 2.0 * tf.cast(input_images, tf.float32) / 255.0
    #input_images.set_shape([(LEFT_CONTEXT+SEQ_LEN) * BATCH_SIZE, HEIGHT, WIDTH, CHANNELS])
    visual_conditions_reshaped = P3D_Vision_Simple(image=input_images, keep_prob=keep_prob, 
                                                     batch_size=BATCH_SIZE, seq_len=SEQ_LEN, scope="P3D", reuse=tf.AUTO_REUSE)

    visual_conditions = tf.reshape(visual_conditions_reshaped, [BATCH_SIZE, SEQ_LEN, -1])
    visual_conditions = tf.nn.dropout(x=visual_conditions, keep_prob=keep_prob)
    
    rnn_inputs_with_ground_truth = (visual_conditions, targets_normalized)
    rnn_inputs_autoregressive = (visual_conditions, tf.zeros(shape=(BATCH_SIZE, SEQ_LEN, OUTPUT_DIM), dtype=tf.float32))
    
    internal_cell = tf.nn.rnn_cell.LSTMCell(num_units=RNN_SIZE, num_proj=RNN_PROJ)
    cell_with_ground_truth = SamplingRNNCell(num_outputs=OUTPUT_DIM, use_ground_truth=True, internal_cell=internal_cell)
    cell_autoregressive = SamplingRNNCell(num_outputs=OUTPUT_DIM, use_ground_truth=False, internal_cell=internal_cell)

    
    def get_initial_state(complex_state_tuple_sizes):
        flat_sizes = tf.contrib.framework.nest.flatten(complex_state_tuple_sizes)
        init_state_flat = [tf.tile(
            multiples=[BATCH_SIZE, 1], 
            input=tf.get_variable("controller_initial_state_%d" % i, initializer=tf.zeros_initializer, shape=([1, s]), dtype=tf.float32))
         for i,s in enumerate(flat_sizes)]
        init_state = tf.contrib.framework.nest.pack_sequence_as(complex_state_tuple_sizes, init_state_flat)
        return init_state
    def deep_copy_initial_state(complex_state_tuple):
        flat_state = tf.contrib.framework.nest.flatten(complex_state_tuple)
        flat_copy = [tf.identity(s) for s in flat_state]
        deep_copy = tf.contrib.framework.nest.pack_sequence_as(complex_state_tuple, flat_copy)
        return deep_copy
    
    controller_initial_state_variables = get_initial_state(cell_autoregressive.state_size)
    controller_initial_state_autoregressive = deep_copy_initial_state(controller_initial_state_variables)
    controller_initial_state_gt = deep_copy_initial_state(controller_initial_state_variables)
    

    with tf.variable_scope("predictor"):
        out_gt, final_state_gt = tf.nn.dynamic_rnn(cell=cell_with_ground_truth, inputs=rnn_inputs_with_ground_truth, 
                          sequence_length=[SEQ_LEN]*BATCH_SIZE, initial_state=None, dtype=tf.float32,
                          swap_memory=True, time_major=False)
    with tf.variable_scope("predictor", reuse=True):
        out_regress, final_state_regress = tf.nn.dynamic_rnn(cell=cell_autoregressive, inputs=rnn_inputs_autoregressive, 
                          sequence_length=[SEQ_LEN]*BATCH_SIZE, initial_state=None, dtype=tf.float32,
                          swap_memory=True, time_major=False)

    return out_gt, final_state_gt, out_regress, final_state_regress
