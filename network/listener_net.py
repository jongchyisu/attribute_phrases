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

class listener_net:

    # Create model
    def __init__(self, args, vocab_length=0, graph=None, log_dir='',
                 img_h=224, img_w=224, channel=3, dataset='trainval'):

        if args.train_img_model:
            self.img1 = tf.placeholder(tf.uint8, [None, img_h, img_w, channel])
            self.img2 = tf.placeholder(tf.uint8, [None, img_h, img_w, channel])
        else:
            self.img1 = tf.placeholder(tf.int32, [None])
            self.img2 = tf.placeholder(tf.int32, [None])
            init_weight = np.load('img_feat/'+args.img_model+'/'+dataset+'.npy')
            init_weight = init_weight.astype(np.float32)
            init_weight = tf.constant(init_weight)
            self.cnn_embeddings = tf.get_variable('embeddings', initializer=init_weight, trainable=False)

        if args.pairwise:
            self.text1 = tf.placeholder(tf.int32, [None, args.max_sent_length])
            self.text2 = tf.placeholder(tf.int32, [None, args.max_sent_length])
            self.text_len = tf.placeholder(tf.int32, [None])
        else:
            self.text1 = tf.placeholder(tf.int32, [None, args.max_sent_length])
            self.text2 = tf.placeholder(tf.int32, [None, args.max_sent_length])
            self.s_len1 = tf.placeholder(tf.int32, [None])
            self.s_len2 = tf.placeholder(tf.int32, [None])

        self.is_training_ph = tf.placeholder(tf.bool, [], name='is_training')

        self.vocab_length = vocab_length
        self.log_dir = log_dir
        self.graph = graph
        self.args = args
        self.mode = self.args.mode

        # default is initializers.xavier_initializer()
        self.initializer = tf.truncated_normal_initializer(stddev=0.01)

    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"
    
    def preprocess(self, img):
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
            raise Exception("model type not supported: {}".format(self.img_model))
        
        # TODO
        #if self.is_training():
        #    img = distort_image(img, height, width, bbox, thread_id)
        return img
    
    def build(self):
        if self.args.train_img_model:
            with tf.variable_scope("preprocess"):
                img1_ = self.preprocess(self.img1)
                img2_ = self.preprocess(self.img2)
        
            model_switcher = {'alexnet': AlexNet.img_alex_feat,  # alexnet_v2
                              'inception_v3': inception_v3,
                              'vgg_16': vgg_16}
            model_func = model_switcher.get(self.args.img_model)
            if self.args.img_model == 'alexnet':
                with tf.variable_scope("Alexnet"):
                    i_fc7_feat1 = model_func(img1_, reuse=False)  # is_training can be added...
                    i_fc7_feat2 = model_func(img2_, reuse=True)
            else: # always set is_training = False for pre-trained CNN model
                i_fc7_feat1 = model_func(img1_, trainable=self.args.train_img_model, is_training=False, reuse=False)
                i_fc7_feat2 = model_func(img2_, trainable=self.args.train_img_model, is_training=False, reuse=True)
        else:
            with tf.variable_scope("img_lookup"):
                i_fc7_feat1 = tf.nn.embedding_lookup(self.cnn_embeddings, self.img1)
                i_fc7_feat2 = tf.nn.embedding_lookup(self.cnn_embeddings, self.img2)

        with tf.variable_scope("img_end_fc") as scope:
            dims = [i_fc7_feat1._shape[1].value] * self.args.img_layer_depth
            dims[-1] = self.args.feat_dim
            self.i_feat1 = layers.stack_fc_layers(i_fc7_feat1, dims, initializer=self.initializer, reuse=False,
                                                     is_training=self.is_training_ph, scope_name="img_end_fc", reg_weight=self.args.regularize)
            self.i_feat2 = layers.stack_fc_layers(i_fc7_feat2, dims, initializer=self.initializer, reuse=True,
                                                     is_training=self.is_training_ph, scope_name="img_end_fc", reg_weight=self.args.regularize)

        with tf.variable_scope("embed_lookup"):
            embed1 = layers.embedding_lookup(self.text1, self.vocab_length, self.args.feat_dim, self.args.word_embed_type, self.args.word_count_thresh)
        with tf.variable_scope("embed_lookup", reuse=True):
            embed2 = layers.embedding_lookup(self.text2, self.vocab_length, self.args.feat_dim, self.args.word_embed_type, self.args.word_count_thresh)
        
        with tf.variable_scope("embed_feat") as scope:
            if self.args.sentence_model == 'cnn' or self.args.sentence_model == 'cnn-rnn':
                embed1_expanded = tf.expand_dims(embed1, -1)
                embed2_expanded = tf.expand_dims(embed2, -1)
                self.w_feat1 = self.embeds_textcnn_feat(embed1_expanded, reuse=False)
                self.w_feat2 = self.embeds_textcnn_feat(embed2_expanded, reuse=True)
            else:
                if self.args.pairwise:
                    self.w_feat1 = self.embeds_rnn_feat(embed1, self.text_len, reuse=False)
                    self.w_feat2 = self.embeds_rnn_feat(embed2, self.text_len, reuse=True)
                else:
                    self.w_feat1 = self.embeds_rnn_feat(embed1, self.s_len1, reuse=False)
                    self.w_feat2 = self.embeds_rnn_feat(embed2, self.s_len2, reuse=True)
        
        self.loss, self.pred_w12, self.prob_w12, self.acc_txt2img_w12, self.pred_i1, self.pred_i2, self.pred_w1, self.pred_w2, \
            self.prob_i1, self.prob_i2, self.prob_w1, self.prob_w2, self.acc_img2txt, self.acc_txt2img = self.loss_sum()

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('acc_img2txt', self.acc_img2txt)
        tf.summary.scalar('acc_txt2img', self.acc_txt2img)

        self.batch = tf.placeholder(tf.uint8, [None])
        with tf.variable_scope("optimizer") as scope:
            self.train = self.train()

        self.merged = tf.summary.merge_all()
        if self.mode == 'train':
            self.train_writer = tf.summary.FileWriter(self.log_dir + '/train', self.graph)
        elif self.mode == 'eval':
            self.val_writer = tf.summary.FileWriter(self.log_dir + '/val')
        else:#self.mode == 'test'
            self.test_writer = tf.summary.FileWriter(self.log_dir + '/test')
        

    # Load checkpoint
    def setup_image_model_initializer(self):
        """Sets up the function to restore inception variables from checkpoint."""
        if self.mode != "inference":
            # Restore inception variables only.
            saver = tf.train.Saver(self.image_model_variables)

            def restore_fn(sess):
                checkpoint_file = 'models/checkpoints/'+self.args.img_model+'.ckpt'
                tf.logging.info("Restoring Inception variables from checkpoint file %s",
                                checkpoint_file)
                saver.restore(sess, checkpoint_file)

            self.init_fn = restore_fn
            
    def embeds_textcnn_feat(self, embed_expanded, dropout=False, reuse=False):
        with tf.variable_scope('embeds_textcnn_feat', reuse=reuse):
            filter_sizes = [3,4,5]
            num_filters = 100
            embedding_size = 300
            dropout_keep_prob = 0.8
            sequence_length = self.args.max_sent_length
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            import pdb
            pdb.set_trace()
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        embed_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            h_pool = tf.concat(axis=3, values=pooled_outputs)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

            # Add dropout
            if dropout:
                with tf.name_scope("dropout"):
                    h_pool_flat = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
            return h_pool_flat
            
    def embeds_rnn_feat(self, embeds, s_len, dropout=False, reuse=False):
        with tf.variable_scope('embeds_rnn_feat', reuse=reuse):
            n_hidden = self.args.feat_dim
            depth = self.args.word_layer_depth
            model = self.args.sentence_model
            state_or_output = self.args.state_or_output

            if model == 'rnn':
                cell_fn = tf.contrib.rnn.BasicRNNCell
            elif model == 'gru':
                cell_fn = tf.contrib.rnn.GRUCell
            elif model == 'lstm':
                cell_fn = tf.contrib.rnn.BasicLSTMCell
            else:
                raise Exception("model type not supported: {}".format(model))

            if self.args.bidirectional:
                # hidden nodes/2 to match the dimension
                fwd_cell = cell_fn(n_hidden/2)
                bwd_cell = cell_fn(n_hidden/2)
                fwd_cell = tf.contrib.rnn.MultiRNNCell([fwd_cell] * depth)
                bwd_cell = tf.contrib.rnn.MultiRNNCell([bwd_cell] * depth)

                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fwd_cell, bwd_cell, embeds, dtype=tf.float32, sequence_length=s_len)

                output_state_fw, output_state_bw = output_states
                output_fw, output_bw = outputs

                if state_or_output == 'state':
                    if model == 'lstm': # use hidden state h for LSTM
                        output = tf.concat(axis=1, values=[output_state_fw[0].h, output_state_bw[0].h])# (?,300)
                    else:
                        output = tf.concat(axis=1, values=[output_state_fw[0], output_state_bw[0]])# (?,300)
                else: # use output
                    output = tf.concat(axis=2, values=[output_fw, output_bw])# (?,10,300)
                    #output = tf.transpose(output, [1, 0, 2])# (10,?,300) don't know why this is wrong

                '''
                ## This is for tf.nn.bidirectional_rnn, now use tf.nn.bidirectional_dynamic_rnn instead
                # Prepare data shape to match `bidirectional_rnn` function requirements
                # Current data input shape: (batch_size, n_steps, n_input)
                # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
                # Permuting batch_size and n_steps
                embeds = tf.transpose(embeds, [1, 0, 2])
                # Reshape to (n_steps*batch_size, n_input)
                embeds = tf.reshape(embeds, [-1, self.args.feat_dim])
                # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
                embeds = tf.split(0, self.args.max_sent_length, embeds)# 0.11
                #x = tf.split(embeds, self.args.max_sent_length, 0)# 0.12

                output, output_state_fw, output_state_bw = tf.nn.bidirectional_rnn(fwd_cell, bwd_cell, embeds, dtype=tf.float32, sequence_length=s_len)
                if state_or_output == 'state':
                    if model == 'lstm': # use hidden state h for LSTM
                        output = tf.concat(1, [output_state_fw[0].h, output_state_bw[0].h])# (?,300)
                    else:
                        output = tf.concat(1, [output_state_fw[0], output_state_bw[0]])# (?,300)
                else: # use output
                    output = tf.concat(0, output)
                    output = tf.reshape(output, [self.args.max_sent_length,-1,self.args.feat_dim])# (10,?,300)
                '''
            else:
                cell = cell_fn(n_hidden)
                cell = tf.contrib.rnn.MultiRNNCell([cell] * depth)
                output, output_state = tf.nn.dynamic_rnn(cell, embeds, dtype=tf.float32, sequence_length=s_len)

                if state_or_output == 'state': # final state
                    if model == 'lstm': # use hidden state h for LSTM
                        output = output_state[0].h# (?,300)
                    else:
                        output = output_state[0]# (?,300)
                else: # use output
                    output = output
                    #output = tf.transpose(output, [1, 0, 2])# (10,?,300) don't know why this is wrong

            if dropout:
                output = tf.nn.dropout(output, keep_prob=0.8)

            if state_or_output == 'output': # if use output, need to choose the last output
                flat = tf.reshape(output, [-1, self.args.feat_dim])# (10*?,300)
                index = tf.range(0, self.args.batch_size) * self.args.max_sent_length + (s_len - 1)
                output = tf.gather(flat, index)

            return output
        
    def _cosine_dis(self, t1, t2):
        prod = tf.reduce_sum(tf.multiply(t1, t2), axis=1)
        norm1 = tf.sqrt(tf.reduce_sum(tf.multiply(t1, t1), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.multiply(t2, t2), axis=1))
        return tf.subtract(1.0, tf.div(prod, tf.multiply(norm1, norm2)))

    def _l2_dis(self, t1, t2):
        return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(t1, t2)), axis=1))
        
    def _l2sq_dis(self, t1, t2):
        return tf.reduce_sum(tf.square(tf.subtract(t1, t2)), axis=1)
    
    def _dotProd_dis(self, t1, t2):
        # return the NEGATIVE of dot-product
        return tf.negative(tf.reduce_sum(tf.multiply(t1, t2), axis=1))

    def _compare_dis(self, query_f, match_f, mismatch_f, distance):
        if distance=='l2sq' or distance=='l2':
            # if distance to negative example is greater than positive example
            dist_pos = self._l2sq_dis(query_f, match_f)
            dist_neg = self._l2sq_dis(query_f, mismatch_f)
            return tf.greater(dist_neg, dist_pos)
        elif distance=='dotProd' or distance=='cosine':
            # if dot product of positive example is greater than negative example
            return tf.greater(tf.reduce_sum(tf.multiply(query_f, match_f), axis=1),
                              tf.reduce_sum(tf.multiply(query_f, mismatch_f), axis=1))

    def _triplet_loss(self, query_f, match_f, mismatch_f, margin=5.0, distance='dotProd'):
        """create the loss function:
        given the feature of query image/embed,
        we want to maximize its product with the matched embed/image
        and minimize its product with the not matched embed/image

        ##loss = max{0, margin - (query_f * match_f - query_f * mismatch_f)}
        loss = softmax_cross_entropy_loss(dis_func(query_f*match_f),dis_func(query_f*mismatch_f))

        :return: loss op
        """
        dis_switcher = {'cosine': self._cosine_dis,
                        'l2': self._l2_dis,
                        'l2sq': self._l2sq_dis,
                        'dotProd': self._dotProd_dis}
        dis_func = dis_switcher.get(distance)
        ''' 
        ##use hinge loss
        score = tf.subtract(dis_func(query_f, mismatch_f), dis_func(query_f, match_f))
        margin_t = tf.fill(tf.shape(score), margin)
        loss = tf.maximum(tf.subtract(margin_t, score), 0.0)
        loss = tf.maximum(tf.subtract(margin_t, tf.sigmoid(score)), 0.0)
        loss = tf.sigmoid(-score)
        '''
        # Now use softmax instead
        correct = -tf.expand_dims(dis_func(query_f, match_f), 1)
        incorrect = -tf.expand_dims(dis_func(query_f, mismatch_f), 1)
        logit = tf.concat([correct, incorrect], axis=1)

        ones = tf.fill(tf.shape(correct), 1.0)
        zeros = tf.fill(tf.shape(correct), 0.0)
        label = tf.concat([ones, zeros], axis=1)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=logit)

        prob = tf.nn.softmax(logits=logit)
        pred = tf.equal(tf.argmax(prob,1), tf.argmax(label,1))

        return tf.reduce_mean(loss), prob, pred, logit
        
    def _contrastive_loss(self, query_f, match_f, mismatch_f, margin, distance):
        ## Not using it now!
        dis_switcher = {'cosine': self._cosine_dis,
                        'l2': self._l2_dis,
                        'l2sq': self._l2sq_dis,
                        'dotProd': self._dotProd_dis}
        dis_func = dis_switcher.get(distance)
        score_neg = dis_func(query_f, mismatch_f)
        score_pos = dis_func(query_f, match_f)
        margin_t = tf.fill(tf.shape(score_neg), margin)
        loss_neg = tf.maximum(tf.subtract(margin_t, score_neg), 0.0)
        return tf.reduce_mean(tf.add(score_pos,loss_neg))

    def loss_sum(self):
        """
        sum up the loss of 4 queries within one set of {img1, embed1, img2, embed2}
        :return: loss op
        """
        loss_switcher = {'contrastive': self._contrastive_loss,
                         'triplet': self._triplet_loss}
        loss_func = loss_switcher.get(self.args.loss_fun)
        loss_i1, prob_i1, pred_i1, _ = loss_func(self.i_feat1, self.w_feat1, self.w_feat2)#, self.args.margin, self.args.distance)
        loss_i2, prob_i2, pred_i2, _ = loss_func(self.i_feat2, self.w_feat2, self.w_feat1)#, self.args.margin, self.args.distance)
        loss_w1, prob_w1, pred_w1, self.logit_w1 = loss_func(self.w_feat1, self.i_feat1, self.i_feat2)#, self.args.margin, self.args.distance)
        loss_w2, prob_w2, pred_w2, self.logit_w2 = loss_func(self.w_feat2, self.i_feat2, self.i_feat1)#, self.args.margin, self.args.distance)
        
        acc_img2txt = (tf.reduce_mean(tf.cast(pred_i1, tf.float32)) +
                       tf.reduce_mean(tf.cast(pred_i2, tf.float32))) /2.0
        acc_txt2img = (tf.reduce_mean(tf.cast(pred_w1, tf.float32)) +
                       tf.reduce_mean(tf.cast(pred_w2, tf.float32))) /2.0

        # average 2 predictions of L
        temp = tf.expand_dims(self._dotProd_dis(self.i_feat1, self.w_feat1), 1)
        ones = tf.fill(tf.shape(temp), 1.0)
        zeros = tf.fill(tf.shape(temp), 0.0)
        label = tf.concat([ones, zeros], axis=1)
        self.prob_w12 = (prob_w1 + prob_w2)/2.0
        self.pred_w12 = tf.equal(tf.argmax(self.prob_w12,1), tf.argmax(label,1))
        self.acc_txt2img_w12 = tf.reduce_mean(tf.cast(self.pred_w12, tf.float32))

        # Regularization
        fc_variables = {v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='img_end_fc')}
        regularize_loss = tf.add_n([tf.nn.l2_loss(v) for v in fc_variables]) * 0.001

        if self.args.loss_type == 'sen':
            return tf.add_n([loss_w1, loss_w2]), self.pred_w12, self.prob_w12, self.acc_txt2img_w12, \
                pred_i1, pred_i2, pred_w1, pred_w2, prob_i1, prob_i2, prob_w1, prob_w2, acc_img2txt, acc_txt2img # only use sentence loss
        elif self.args.loss_type == 'img':
            return tf.add_n([loss_i1, loss_i2]), \
                pred_i1, pred_i2, pred_w1, pred_w2, prob_i1, prob_i2, prob_w1, prob_w2, acc_img2txt, acc_txt2img # only use image loss
        elif self.args.loss_type == 'all':
            return tf.add_n([loss_i1, loss_i2, loss_w1, loss_w2]), \
                pred_i1, pred_i2, pred_w1, pred_w2, prob_i1, prob_i2, prob_w1, prob_w2, acc_img2txt, acc_txt2img
        else:
            raise Exception("loss type not supported: {}".format(self.args.loss_type))

    def train(self):
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learn_rate, beta1=self.args.beta1)  #, epsilon=self.epsilon)

        train_op = optimizer.minimize(self.loss)
        return train_op

