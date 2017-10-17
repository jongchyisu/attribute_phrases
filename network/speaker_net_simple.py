import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf
import sys

from network import layers as my_layers
from models import AlexNet
from models.Alexnet_slim import alexnet_v2
from models.Inception_v3 import inception_v3
from models.vgg_net_slim import vgg_16


class CaptionNetSingle:
    def __init__(self, args, mode, img_h=224, img_w=224, img_channel=3, vocab_size=0):
        assert args.img_model == 'vgg_16'
        assert args.img_input_mode == 'per_state'
        assert args.img_end_layers == [1024, 512, -1]
        assert args.rnn_cell == 'lstm'
        assert args.rnn_depth == 1

        self.vocab_size = vocab_size
        self.args = args
        self.mode = mode

        self.initializer = tf.truncated_normal_initializer(stddev=0.01)

        if args.train_img_model:
            self.img1 = tf.placeholder(tf.uint8, [None, img_h, img_w, img_channel], name='image1')
        else:
            self.img1 = tf.placeholder(tf.int32, [None], name='image1')

        if mode == 'inference':
            self.text = tf.placeholder(tf.int32, shape=[None], name='text_seq')
            self.text = tf.expand_dims(self.text, 1)
        else:
            self.text = tf.placeholder(tf.int32, [None, args.max_text_length])
            self.target_seq = tf.placeholder(tf.int32, [None, args.max_text_length])
            self.text_mask = tf.placeholder(tf.int32, [None, args.max_text_length])

        self.img_embed = None
        self.text_embeds = None
        self.img_text_embed_seq = None

        self.batch_loss = None
        self.target_cross_entropy_losses = None
        self.target_cross_entropy_loss_weights = None

        # self.global_step = None
        self.train = None
        self.merged_summary = None

    def _preprocess(self, img):
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        if self.args.img_model == 'inception_v3':
            # rescale to [-1,1] instead of [0, 1)
            img = np.subtract(img, 0.5)
            img = np.multiply(img, 2.0)
        elif self.args.img_model == 'alexnet' or self.args.img_model == 'vgg_16':
            # zero-mean input
            img = np.multiply(img, 255.0)
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            img = img - mean
        else:
            raise Exception("model type not supported: {}".format(self.args.img_model))

        # TODO
        # if self.is_training():
        #    img = distort_image(img, height, width, bbox, thread_id)
        return img

    def build_input(self, img_end_layers, word_embed_dim, word_embed_type):
        if self.args.train_img_model:
            with tf.variable_scope("img_preprocess"):
                img1_ = self._preprocess(self.img1)
            i_fc7_feat1 = vgg_16(img1_, trainable=self.args.train_img_model, is_training=False, reuse=False)
        else:
            with tf.variable_scope("img_lookup"):
                init_weight = np.load('../cnn_feat/%s/%s.npy' %(self.args.img_model, self.args.train_set_name))
                init_weight = init_weight.astype(np.float32)
                init_weight = tf.constant(init_weight)
                cnn_embeddings = tf.get_variable('embeddings', initializer=init_weight, trainable=False)
                i_fc7_feat1 = tf.nn.embedding_lookup(cnn_embeddings, self.img1)

        with tf.variable_scope("img_end_fc"):
            stop_idx = img_end_layers.index(-1)
            fc_dims = img_end_layers[0:stop_idx]
            i_feat = my_layers.stack_fc_layers(i_fc7_feat1, fc_dims, scope_name='sep_fc', reuse=False,
                                               initializer=self.initializer, is_training=self._is_training())
            self.img_embed = tf.identity(i_feat, name='img_embed')

        with tf.variable_scope("embed_lookup"):
            self.text_embeds = my_layers.embedding_lookup(self.text, self.vocab_size, word_embed_dim,
                                                          word_embed_type, self.args.word_count_thresh, '../models')

        tile_img_embeds = tf.expand_dims(self.img_embed, 1)
        tile_img_embeds = tf.tile(tile_img_embeds, [1, self.text_embeds.get_shape()[1].value, 1])
        self.img_text_embed_seq = tf.concat(axis=2, values=[tile_img_embeds, self.text_embeds])

    def build_rnn(self, rnn_num_units, dropout_keep_prob):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_num_units, state_is_tuple=True)
        if self.mode == "train":
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=dropout_keep_prob,
                                                      output_keep_prob=dropout_keep_prob)

        with tf.variable_scope("rnn", initializer=self.initializer):
            zeros_dims = tf.stack([tf.shape(self.img_embed)[0], self.args.word_embed_dim])
            zero_embed = tf.zeros(zeros_dims)
            init_input = tf.concat(axis=1, values=[self.img_embed, zero_embed])
            rnn_input_seq = self.img_text_embed_seq

            # Feed the image embeddings to set the initial rnn state.
            zero_state = lstm_cell.zero_state(batch_size=self.args.batch_size, dtype=tf.float32)
            _, initial_state = lstm_cell(inputs=init_input, state=zero_state)

        with tf.variable_scope("rnn", reuse=True) as rnn_scope:
            if self.mode == "train":
                # Run the batch of sequence embeddings through the rnn.
                text_valid_len = tf.reduce_sum(self.text_mask, 1)
                # print 'init_input: %d' % tf.shape(init_input)[1].value
                # print 'init_state: %d, %d' % (tf.shape(initial_state)[0].value, tf.shape(initial_state)[1].value)
                rnn_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=rnn_input_seq,
                                                   sequence_length=text_valid_len,
                                                   initial_state=initial_state, dtype=tf.float32, scope=rnn_scope)
            else:  # inference
                # In inference mode, use concatenated states for convenient feeding and fetching.
                # Placeholder for feeding a batch of concatenated states.
                tf.concat(axis=1, values=initial_state, name="initial_state")
                state_feed = tf.placeholder(dtype=tf.float32, shape=[None, sum(lstm_cell.state_size)],
                                            name="state_feed")
                state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)
                # Run a single RNN step.
                rnn_outputs, state_tuple = lstm_cell(inputs=tf.squeeze(rnn_input_seq, axis=[1]),
                                                    state=state_tuple)
                # Concatenate the resulting state.
                tf.concat(axis=1, values=state_tuple, name="state")

        # Stack batches vertically.
        rnn_outputs = tf.reshape(rnn_outputs, [-1, lstm_cell.output_size])
        return rnn_outputs

    def rnn_end(self, rnn_outputs):
        with tf.variable_scope("logits") as logits_scope:
            logits = slim.fully_connected(inputs=rnn_outputs, num_outputs=self.vocab_size, activation_fn=None,
                                          weights_initializer=self.initializer, scope=logits_scope)
        if self.mode == "inference":
            tf.nn.softmax(logits, name="softmax")
        else:
            targets = tf.reshape(self.target_seq, [-1])
            weights = tf.to_float(tf.reshape(self.text_mask, [-1]))
            # Compute losses.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
            batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)), tf.reduce_sum(weights), name="batch_loss")

            self.batch_loss = batch_loss
            self.target_cross_entropy_losses = losses  # Used in evaluation.
            self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

            # for var in tf.trainable_variables():
            #     tf.summary.histogram("parameters/" + var.op.name, var)

    def build(self):
        self.build_input(self.args.img_end_layers, self.args.word_embed_dim, self.args.word_embed_type)
        rnn_outputs = self.build_rnn(self.args.rnn_num_units, self.args.dropout_keep_prob)
        self.rnn_end(rnn_outputs)
        if self.mode is not 'inference':
            with tf.variable_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learn_rate, beta1=self.args.adam_beta1,
                                                   epsilon=self.args.adam_epsilon)
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.args.learn_rate)
                self.train = optimizer.minimize(self.batch_loss)

            tf.summary.scalar('batch_loss', self.batch_loss)
            self.merged_summary = tf.summary.merge_all()

    def _is_training(self):
        return self.mode == 'train'

