import tensorflow as tf
import P3D
import T3D
import ResNet
from ResNet import NUM_FEAT
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


def inference_att(input_images, targets_normalized, keep_prob):

    weight_initializer = tf.contrib.layers.xavier_initializer()
    const_initializer = tf.constant_initializer(0.0)

    def _get_initial_lstm(features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [VISION_FEATURE_SIZE, RNN_SIZE], initializer=weight_initializer)
            b_h = tf.get_variable('b_h', [RNN_SIZE], initializer=const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [VISION_FEATURE_SIZE, RNN_SIZE], initializer=weight_initializer)
            b_c = tf.get_variable('b_c', [RNN_SIZE], initializer=const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _get_initial_targets(features_mean):
        with tf.variable_scope('initial_target'):
            w_h = tf.get_variable('w_out', [RNN_SIZE, OUTPUT_DIM], initializer=weight_initializer)
            b_h = tf.get_variable('b_out', [OUTPUT_DIM], initializer=const_initializer)
            target = tf.matmul(features_mean, w_h) + b_h
            return target

    def _project_features(features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [VISION_FEATURE_SIZE, VISION_FEATURE_SIZE], initializer=weight_initializer)
            features_flat = tf.reshape(features, [-1, VISION_FEATURE_SIZE])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, NUM_FEAT, VISION_FEATURE_SIZE])
            return features_proj

    def _attention_layer(features, features_proj, h):
        with tf.variable_scope('attention_layer'):
            w = tf.get_variable('w', [RNN_SIZE, VISION_FEATURE_SIZE], initializer=weight_initializer)
            b = tf.get_variable('b', [VISION_FEATURE_SIZE], initializer=const_initializer)
            w_att = tf.get_variable('w_att', [VISION_FEATURE_SIZE, 1], initializer=weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, VISION_FEATURE_SIZE]), w_att), [-1, NUM_FEAT])   # (N, L)
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
            return context, alpha

    def _decode_lstm(x, h, context, dropout=False):
        with tf.variable_scope('logits'):
            expanded_output = tf.concat([x,
                                        h,
                                        context],
                                        axis = 1)
            w_out = tf.get_variable('w_out', [OUTPUT_DIM+RNN_SIZE+VISION_FEATURE_SIZE, OUTPUT_DIM], initializer=weight_initializer)
            b_out = tf.get_variable('b_out', [OUTPUT_DIM], initializer=const_initializer)
            
            out_logits = tf.matmul(expanded_output, w_out) + b_out
            return out_logits

    input_images = -1.0 + 2.0 * tf.cast(input_images, tf.float32) / 255.0
    input_images  = tf.reshape(input_images, shape=[-1, LEFT_CONTEXT + SEQ_LEN, HEIGHT, WIDTH, RGB_CHANNEL])
    print('input_shape: ', input_images.shape.as_list())


    c = h = None
    internal_cell = tf.nn.rnn_cell.LSTMCell(num_units=RNN_SIZE, num_proj=RNN_PROJ)

    logits = []
    for idx in range(SEQ_LEN):
        with tf.variable_scope('RNN', reuse=(idx!=0)):
            # visual feature
            with tf.variable_scope('Vision'):
                visual_conditions = ResNet.inference(input_images[:,idx+1:LEFT_CONTEXT+idx+1,:,:,:],VISION_FEATURE_SIZE,keep_prob)
                print('vfeat_shape: ', visual_conditions.shape.as_list())
            
            # initialize the LSTM
            if idx == 0:
                c, h = _get_initial_lstm(features=visual_conditions)

            # project visual feature
            features_proj = _project_features(features=visual_conditions)

            # attend machenism
            context, alpha = _attention_layer(visual_conditions, features_proj, h)

            # LSTM unit
            with tf.variable_scope('lstm'):
                _, (c, h) = internal_cell(inputs=tf.concat( [targets_normalized[:,idx,:], context],1), state=[c, h])

            output = _decode_lstm(targets_normalized[:,idx,:], h, context, dropout=keep_prob)
            logits.append(tf.expand_dims(output,1))
    logits = tf.concat(logits, axis=1)

    print('output_shape: ', logits.shape.as_list())
    return logits


def inference(input_images, targets_normalized, keep_prob):
    input_images = -1.0 + 2.0 * tf.cast(input_images, tf.float32) / 255.0
    #input_images.set_shape([(LEFT_CONTEXT+SEQ_LEN) * BATCH_SIZE, HEIGHT, WIDTH, CHANNELS])
    print('input_shape: ', input_images.shape.as_list())

    input_images = tf.reshape(input_images, shape=[-1, LEFT_CONTEXT + SEQ_LEN, HEIGHT, WIDTH, RGB_CHANNEL])
    visual_conditions = T3D_Vision_Simple(inputs=input_images, keep_prob=keep_prob, 
                                                     seq_len=SEQ_LEN, scope="Vision", reuse=tf.AUTO_REUSE)

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
