import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os


def embedding_lookup(text, V, K, word_embed_type='one-hot', word_count_thresh=5, models_folder='models'):
    """
    Given the text, create a embedding lookup
    Args:
        text: (tf.placeholder) the input tokens sequence
        V: size of vocabulary
        K: size of embeddings
        word2vec: (boolean) whether or not initialize with pre-trained word2vec
    Returns:
        embed: (Tensor) B x maxlen x K tensor of embeddings
    """

    if word_embed_type == 'word2vec':
        init_weight = np.load(os.path.join(models_folder, 'word2vec_concise_'+str(word_count_thresh)+'.npy'))
        init_weight = init_weight.astype(np.float32)
        init_weight = tf.constant(init_weight)
        embeddings = tf.get_variable('embeddings', initializer=init_weight, trainable=False)
    elif word_embed_type == 'one-hot':
        init_weight = tf.random_uniform([V, K], -.001, .001, dtype=tf.float32)
        embeddings = tf.get_variable('embeddings', initializer=init_weight, trainable=True)
    elif word_embed_type == 'char':
        init_weight = tf.random_uniform([V, K], -.001, .001, dtype=tf.float32)
        embeddings = tf.get_variable('embeddings', initializer=init_weight, trainable=True)
    else:
        raise Exception("model type not supported: {}".format(word_embed_type))

    embed = tf.nn.embedding_lookup(embeddings, text)
    return embed


def stack_fc_layers(input_tensor, dims, initializer, is_training, scope_name, reuse, reg_weight=0.0, trainable=True):
    fc_feat = input_tensor
    with slim.arg_scope([slim.fully_connected], weights_initializer=initializer, reuse=reuse, trainable=trainable):
        with slim.arg_scope([slim.batch_norm], is_training=is_training, updates_collections=None):
            i = 0
            for dim in dims:
                fc_feat = slim.fully_connected(inputs=fc_feat, num_outputs=int(dim),
                                               weights_regularizer=slim.l2_regularizer(reg_weight),
                                               normalizer_fn=slim.batch_norm, scope=scope_name + str(i))
                i += 1
    return fc_feat
