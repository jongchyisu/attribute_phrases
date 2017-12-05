import tensorflow as tf
import numpy as np
import math


def _conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    """ From https://github.com/ethereon/caffe-tensorflow
    Helper function to create one conv layer
    """
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(axis=3, num_or_size_splits=group, value=input)
        kernel_groups = tf.split(axis=3, num_or_size_splits=group, value=kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(axis=3, values=output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])


def _prod(dims):
    """ Helper function
    """
    p = 1
    for d in dims:
        p *= d
    return p


def img_alex_feat(img, init_weight_path='../model/bvlc_alexnet.npy', reuse=False):
    """Build convolution followed by fully connected network

    get AlexNet fc7 feature for images
    :arg:
        x: input images
        init_weight_path: load pre-trained network
    :return: fcc op
    """
    with tf.variable_scope('AlexNet', reuse=reuse):
        # init weight
        if init_weight_path is None:
            init_weight = None
        else:
            init_weight = np.load(init_weight_path).item()

        # conv1
        k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4; group = 1
        if init_weight is not None:
            init_conv1w = init_weight['conv1'][0]
            init_conv1b = init_weight['conv1'][1]
        else:
            init_conv1w = tf.truncated_normal([k_h, k_w, int(img.get_shape().as_list()[3] / group), c_o],
                                              stddev=0.05)
            init_conv1b = tf.constant(0.01, shape=[c_o])

        conv1w = tf.get_variable('conv1w', initializer=init_conv1w)
        conv1b = tf.get_variable('conv1b', initializer=init_conv1b)
        conv1_in = _conv(img, conv1w, conv1b, k_h, k_w, c_o, s_h, s_w, padding='SAME', group=group)
        conv1 = tf.nn.relu(conv1_in, name='conv1')

        # lrn1
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

        # maxpool1
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool1')

        # # conv2
        k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group=2

        if init_weight is not None:
            init_conv2w = init_weight['conv2'][0]
            init_conv2b = init_weight['conv2'][1]
        else:
            init_conv2w = tf.truncated_normal([k_h, k_w, int(maxpool1.get_shape().as_list()[3]/group), c_o],
                                              stddev=0.05)
            init_conv2b = tf.constant(0.01, shape=[c_o])

        conv2w = tf.get_variable('conv2w', initializer=init_conv2w)
        conv2b = tf.get_variable('conv2b',initializer=init_conv2b)
        conv2_in = _conv(maxpool1, conv2w, conv2b, k_h, k_w, c_o, s_h, s_w, padding='SAME', group=group)
        conv2 = tf.nn.relu(conv2_in, name='conv2')

        # lrn2
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

        # maxpool2
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool2')

        # conv3
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
        if init_weight is not None:
            init_conv3w = init_weight['conv3'][0]
            init_conv3b = init_weight['conv3'][1]
        else:
            init_conv3w = tf.truncated_normal([k_h, k_w, int(maxpool2.get_shape().as_list()[3]/group), c_o],
                                              stddev=0.05)
            init_conv3b = tf.constant(0.01, shape=[c_o])

        conv3w = tf.get_variable('conv3w', initializer=init_conv3w)
        conv3b = tf.get_variable('conv3b', initializer=init_conv3b)
        conv3_in = _conv(maxpool2, conv3w, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group = group)
        conv3 = tf.nn.relu(conv3_in, name='conv3')

        # conv4
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
        if init_weight is not None:
            init_conv4w = init_weight['conv4'][0]
            init_conv4b = init_weight['conv4'][1]
        else:
            init_conv4w = tf.truncated_normal([k_h, k_w, int(conv3.get_shape().as_list()[3]/group), c_o],
                                              stddev=0.05)
            init_conv4b = tf.constant(0.01, shape=[c_o])

        conv4w = tf.get_variable('conv4w', initializer=init_conv4w)
        conv4b = tf.get_variable('conv4b', initializer=init_conv4b)
        conv4_in = _conv(conv3, conv4w, conv4b, k_h, k_w, c_o, s_h, s_w, padding='SAME', group=group)
        conv4 = tf.nn.relu(conv4_in, name='conv4')

        # conv5
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
        if init_weight is not None:
            init_conv5w = init_weight['conv5'][0]
            init_conv5b = init_weight['conv5'][1]
        else:
            init_conv5w = tf.truncated_normal([k_h, k_w, int(conv4.get_shape().as_list()[3]/group), c_o],
                                              stddev=0.05)
            init_conv5b = tf.constant(0.01, shape=[c_o])

        conv5w = tf.get_variable('conv5w', initializer=init_conv5w)
        conv5b = tf.get_variable('conv5b', initializer=init_conv5b)
        conv5_in = _conv(conv4, conv5w, conv5b, k_h, k_w, c_o, s_h, s_w, padding='SAME', group=group)
        conv5 = tf.nn.relu(conv5_in, name='conv5')

        # maxpool5
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1],
                                  padding=padding, name='pool5')

        # fc6
        c_o = 4096
        fc_input_dim = int(_prod(maxpool5.get_shape().as_list()[1:]))

        if init_weight is not None:
            init_fc6w = init_weight['fc6'][0]
            init_fc6b = init_weight['fc6'][1]
        else:
            init_fc6w = tf.truncated_normal([fc_input_dim, c_o], stddev=1.0/math.sqrt(float(fc_input_dim)))
            init_fc6b = tf.zeros([c_o])

        fc6w = tf.get_variable('fc6w', initializer=init_fc6w)
        fc6b = tf.get_variable('fc6b', initializer=init_fc6b)
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, fc_input_dim]), fc6w, fc6b, name='fc6')

        # fc7
        c_o = 4096
        fc_input_dim = fc6.get_shape().as_list()[1]
        if init_weight is not None:
            init_fc7w = init_weight['fc7'][0]
            init_fc7b = init_weight['fc7'][1]
        else:
            init_fc7w = tf.truncated_normal([fc_input_dim, c_o], stddev=1.0/math.sqrt(float(fc_input_dim)))
            init_fc7b = tf.zeros([c_o])

        fc7w = tf.get_variable('fc7w', initializer=init_fc7w)
        fc7b = tf.get_variable('fc7b', initializer=init_fc7b)
        fc7 = tf.nn.relu_layer(fc6, fc7w, fc7b, name='fc7')

        '''
        # fc8
        c_o = self.feat_len  # change class numbers here!!!
        fc_input_dim = fc7.get_shape().as_list()[1]
        # if init_weight is not None:
        #     init_fc8w = init_weight['fc8'][0]
        #     init_fc8b = init_weight['fc8'][1]
        # else:
        init_fc8w = tf.truncated_normal([fc_input_dim, c_o], stddev=1.0/math.sqrt(float(fc_input_dim)))
        init_fc8b = tf.zeros([c_o])

        fc8w = tf.get_variable('fc8w', initializer=init_fc8w)
        fc8b = tf.get_variable('fc8b', initializer=init_fc8b)
        fc8 = tf.nn.xw_plus_b(fc7, fc8w, fc8b, name='fc8')
        '''
        return fc7
