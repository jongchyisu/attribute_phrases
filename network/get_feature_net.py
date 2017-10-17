from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
from network import layers
from models import AlexNet
from models.Alexnet_slim import alexnet_v2
from models.Inception_v3 import inception_v3
from models.vgg_net_slim import vgg_16

class image_net:
    # Create model
    def __init__(self, args, graph=None, img_h=224, img_w=224, channel=3):
        self.img = tf.placeholder(tf.uint8, [1, img_h, img_w, channel])
        self.args = args
        self.mode = self.args.mode
        
    def preprocess(self, img):
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        if self.args.img_model == 'inception_v3':
            # rescale to [-1,1] instead of [0, 1)
            img = np.subtract(img, 0.5)
            img = np.multiply(img, 2.0)
        elif self.args.img_model == 'alexnet' or self.args.img_model == 'vgg_16':
            # zero-mean input
            img = np.multiply(img, 255.0)
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 3], name='img_mean')
            img = img - mean
        else:
            raise Exception("model type not supported: {}".format(self.img_model))
        return img
    
    def build(self):
        with tf.variable_scope("preprocess") as scope:
            self.img_ = self.preprocess(self.img)
        
        model_switcher = {'alexnet': AlexNet.img_alex_feat,
                          'inception_v3': inception_v3,
                          'vgg_16': vgg_16}
        model_func = model_switcher.get(self.args.img_model)
        
        if self.args.img_model == 'alexnet':
            self.img_feat = model_func(self.img_)
        else:
            self.img_feat = model_func(self.img_, trainable=False, is_training=False)

