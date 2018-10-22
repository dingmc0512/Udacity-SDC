import tensorflow as tf
import P3D
import T3D
import ResNet
import STConv
from SamplingRNNCell import SamplingRNNCell
from Pseudo3DConv import p_conv3d
from settings import *
slim = tf.contrib.slim


layer_norm = lambda x: tf.contrib.layers.layer_norm(inputs=x, center=True, scale=True, activation_fn=None, trainable=True)

def ST_Conv_Global_Vision_Simple(inputs, keep_prob, seq_len, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Vision', [inputs], reuse=reuse):
        return STConv.inference(inputs, keep_prob, seq_len)


def ST_Conv_Local_Vision_Simple(inputs, keep_prob, seq_len, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Vision', [inputs], reuse=reuse):
        seq_net = []
        for idx in range(seq_len):
            net = STConv.inference(inputs[:,LEFT_CONTEXT+idx,:,:,:], VISION_FEATURE_SIZE, keep_prob)
            seq_net.append(net)
        return tf.stack(seq_net, axis=1)


def CNN_Vision_Simple(inputs, keep_prob, seq_len, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Vision', [inputs], reuse=reuse):
        seq_net = []
        for idx in range(seq_len):
            net = P3D.inference_p3d(inputs[:,LEFT_CONTEXT+idx,:,:,:],VISION_FEATURE_SIZE,keep_prob)
            seq_net.append(net)
        return tf.stack(seq_net, axis=1)


def P3D_Vision_Simple(inputs, keep_prob, seq_len, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Vision', [inputs], reuse=reuse):
        seq_net = []
        for idx in range(seq_len):
            net = P3D.inference_p3d(inputs[:,idx+1:LEFT_CONTEXT+idx+1,:,:,:],VISION_FEATURE_SIZE,keep_prob)
            seq_net.append(net)
        return tf.stack(seq_net, axis=1)


def T3D_Vision_Simple(inputs, keep_prob, seq_len, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Vision', [inputs], reuse=reuse):
        seq_net = []
        for idx in range(seq_len):
            net = T3D.inference_t3d(inputs[:,idx+1:LEFT_CONTEXT+idx+1,:,:,:],VISION_FEATURE_SIZE,keep_prob)
            seq_net.append(net)
        return tf.stack(seq_net, axis=1)


def RetNet_Vision_Simple(inputs, keep_prob, seq_len, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Vision', [inputs], reuse=reuse):
        seq_net = []
        for idx in range(seq_len):
            net = ResNet.inference(inputs[:,idx+1:LEFT_CONTEXT+idx+1,:,:,:],VISION_FEATURE_SIZE,keep_prob)
            seq_net.append(net)
        return tf.stack(seq_net, axis=1)


def inference(input_images, targets_normalized, keep_prob):
    input_images = -1.0 + 2.0 * tf.cast(input_images, tf.float32) / 255.0
    #input_images.set_shape([(LEFT_CONTEXT+SEQ_LEN) * BATCH_SIZE, HEIGHT, WIDTH, CHANNELS])
    print('input_shape: ', input_images.shape.as_list())

    input_images = tf.reshape(input_images, shape=[-1, LEFT_CONTEXT + SEQ_LEN, HEIGHT, WIDTH, RGB_CHANNEL])
    visual_conditions = RetNet_Vision_Simple(inputs=input_images, keep_prob=keep_prob, 
                                                     seq_len=SEQ_LEN, scope="P3D", reuse=tf.AUTO_REUSE)

    print('vfeat_shape: ', visual_conditions.shape.as_list())
    visual_conditions = tf.reshape(visual_conditions, [BATCH_SIZE, SEQ_LEN, -1])
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

    print('output_shape: ', out_regress.shape.as_list())
    return out_gt, final_state_gt, out_regress, final_state_regress
