import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf

from network import layers as my_layers
from models import AlexNet
from models.Alexnet_slim import alexnet_v2
from models.Inception_v3 import inception_v3
from models.vgg_net_slim import vgg_16


class CaptionNet:
    def __init__(self, args, mode, img_h=224, img_w=224, img_channel=3, vocab_size=0):

        self.vocab_size = vocab_size
        self.args = args
        self.mode = mode

        self.initializer = tf.truncated_normal_initializer(stddev=0.01)

        if args.train_img_model:
            self.img1 = tf.placeholder(tf.uint8, [None, img_h, img_w, img_channel], name='image1')
            self.img2 = tf.placeholder(tf.uint8, [None, img_h, img_w, img_channel], name='image2')
        else:
            self.img1 = tf.placeholder(tf.int32, [None], name='image1')
            self.img2 = tf.placeholder(tf.int32, [None], name='image2')

        if mode == 'inference':
            self.text = tf.placeholder(tf.int32, shape=[None], name='text_seq')
            self.text = tf.expand_dims(self.text, 1)
        else:
            self.text = tf.placeholder(tf.int32, [None, args.max_text_length])
            self.target_seq = tf.placeholder(tf.int32, [None, args.max_text_length])
            self.text_mask = tf.placeholder(tf.int32, [None, args.max_text_length])

        self.img_embed = None
        self.text_embeds = None
        if args.img_input_mode == 'per_state':
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
            model_switcher = {'alexnet': AlexNet.img_alex_feat,  # alexnet_v2
                              'inception_v3': inception_v3,
                              'vgg_16': vgg_16}
            model_func = model_switcher.get(self.args.img_model)

            with tf.variable_scope("img_preprocess"):
                img1_ = self._preprocess(self.img1)
                img2_ = self._preprocess(self.img2)

            if self.args.img_model == 'alexnet':
                with tf.variable_scope("Alexnet"):
                    # is_training can be added...
                    i_fc7_feat1 = model_func(img1_, init_weight_path='../../model/bvlc_alexnet.npy', reuse=False)
                    i_fc7_feat2 = model_func(img2_, init_weight_path='../../model/bvlc_alexnet.npy', reuse=True)
            else:
                i_fc7_feat1 = model_func(img1_, trainable=self.args.train_img_model, is_training=False, reuse=False)
                i_fc7_feat2 = model_func(img2_, trainable=self.args.train_img_model, is_training=False, reuse=True)
        else:
            with tf.variable_scope("img_lookup"):
                init_weight = np.load('../cnn_feat/%s/%s.npy' % (self.args.img_model, self.args.train_set_name))
                init_weight = init_weight.astype(np.float32)
                init_weight = tf.constant(init_weight)
                cnn_embeddings = tf.get_variable('embeddings', initializer=init_weight, trainable=False)
                i_fc7_feat1 = tf.nn.embedding_lookup(cnn_embeddings, self.img1)
                i_fc7_feat2 = tf.nn.embedding_lookup(cnn_embeddings, self.img2)

        with tf.variable_scope("img_end_fc"):
            concat_layer = img_end_layers.index(-1)
            if concat_layer != 0:
                sep_dims = img_end_layers[0:concat_layer]
                print 'sep_dims: ', sep_dims
                i_feat1 = my_layers.stack_fc_layers(i_fc7_feat1, sep_dims, scope_name='sep_fc', reuse=False,
                                                    initializer=self.initializer, is_training=self._is_training())
                i_feat2 = my_layers.stack_fc_layers(i_fc7_feat2, sep_dims, scope_name='sep_fc', reuse=True,
                                                    initializer=self.initializer, is_training=self._is_training())
            else:
                i_feat1 = i_fc7_feat1
                i_feat2 = i_fc7_feat2
            i_feat = tf.concat(axis=1, values=[i_feat1, i_feat2], name='img_embed')
            if img_end_layers[-1] != -1:
                joint_dims = img_end_layers[concat_layer+1:]
                print 'joint_dims: ', joint_dims
                self.img_embed = my_layers.stack_fc_layers(i_feat, joint_dims, scope_name='joint_fc', reuse=False,
                                                           initializer=self.initializer,
                                                           is_training=self._is_training())
            else:
                self.img_embed = i_feat

        with tf.variable_scope("embed_lookup"):
            self.text_embeds = my_layers.embedding_lookup(self.text, self.vocab_size, word_embed_dim,
                                                          word_embed_type, self.args.word_count_thresh, '../models')

        if self.args.img_input_mode == 'per_state':
            tile_img_embeds = tf.expand_dims(self.img_embed, 1)
            tile_img_embeds = tf.tile(tile_img_embeds, [1, self.text_embeds.get_shape()[1].value, 1])
            self.img_text_embed_seq = tf.concat(axis=2, values=[tile_img_embeds, self.text_embeds])

    def build_rnn(self, img_input_mode, rnn_cell_name, rnn_num_units, rnn_depth, dropout_keep_prob):
        rnn_cell_dict = {'rnn': tf.contrib.rnn.BasicRNNCell,
                         'lstm': tf.contrib.rnn.BasicLSTMCell,
                         'gru': tf.contrib.rnn.GRUCell}
        rnn_cell_class = rnn_cell_dict.get(rnn_cell_name)
        if rnn_cell_name == 'lstm':
            rnn_cell = rnn_cell_class(rnn_num_units, state_is_tuple=True)
        else:
            rnn_cell = rnn_cell_class(rnn_num_units)
        if self.mode == "train":
            rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, input_keep_prob=dropout_keep_prob,
                                                     output_keep_prob=dropout_keep_prob)
        if rnn_depth > 1:
            rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell] * rnn_depth)

        with tf.variable_scope("rnn", initializer=self.initializer):

            if img_input_mode == 'initial':
                rnn_input_seq = self.text_embeds
                init_input = self.img_embed
            else:  # 'per_state'
                zeros_dims = tf.stack([tf.shape(self.img_embed)[0], self.args.word_embed_dim])
                zero_embed = tf.zeros(zeros_dims)
                init_input = tf.concat(axis=1, values=[self.img_embed, zero_embed])
                rnn_input_seq = self.img_text_embed_seq

            # Feed the image embeddings to set the initial rnn state.
            zero_state = rnn_cell.zero_state(batch_size=self.args.batch_size, dtype=tf.float32)
            _, initial_state = rnn_cell(inputs=init_input, state=zero_state)

        with tf.variable_scope("rnn", reuse=True) as rnn_scope:
            if self.mode == "train":
                # Run the batch of sequence embeddings through the rnn.
                text_valid_len = tf.reduce_sum(self.text_mask, 1)
                # print 'init_input: %d' % tf.shape(init_input)[1].value
                # print 'init_state: %d, %d' % (tf.shape(initial_state)[0].value, tf.shape(initial_state)[1].value)
                rnn_outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=rnn_input_seq,
                                                   sequence_length=text_valid_len,
                                                   initial_state=initial_state, dtype=tf.float32, scope=rnn_scope)
            else:  # inference
                # In inference mode, use concatenated states for convenient feeding and fetching.
                # Placeholder for feeding a batch of concatenated states.
                if rnn_cell_name == 'lstm':
                    tf.concat(axis=1, values=initial_state, name="initial_state")
                    state_feed = tf.placeholder(dtype=tf.float32, shape=[None, sum(rnn_cell.state_size)],
                                                name="state_feed")
                    state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)
                    # Run a single RNN step.
                    rnn_outputs, state_tuple = rnn_cell(inputs=tf.squeeze(rnn_input_seq, axis=[1]),
                                                        state=state_tuple)
                    # Concatenate the resulting state.
                    tf.concat(axis=1, values=state_tuple, name="state")
                else:
                    initial_state = tf.identity(initial_state, name='initial_state')
                    state_feed = tf.placeholder(dtype=tf.float32, shape=[None, rnn_cell.state_size], name="state_feed")
                    # Run a single RNN step.
                    rnn_outputs, state = rnn_cell(inputs=tf.squeeze(rnn_input_seq, axis=[1]), state=state_feed)
                    # Concatenate the resulting state.
                    state = tf.identity(state_feed, name="state")

        # Stack batches vertically.
        rnn_outputs = tf.reshape(rnn_outputs, [-1, rnn_cell.output_size])
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
        rnn_outputs = self.build_rnn(self.args.img_input_mode, self.args.rnn_cell,
                                     self.args.rnn_num_units, self.args.rnn_depth, self.args.dropout_keep_prob)
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


