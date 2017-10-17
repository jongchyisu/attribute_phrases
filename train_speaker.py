import numpy as np
import time
import os
import argparse
import json
import tensorflow as tf
import sys

from pyVisDifftools.visdiff import VisDiff
import network.speaker_net_discerning as pair_net
import network.speaker_net_simple as single_net
import data_loader as data_feeder
# from inference import inference

# set path
data_dir = 'dataset'
img_dir = 'dataset/images'


def main():
    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument('--speaker_mode', type=str, default='S',
                        help='"DS" for pair speaker, "S" for single speaker')
    parser.add_argument('--train_set_name', type=str, default='train',
                        help='which set to use for training: train/trainval')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='64 for fixCNN, 32 for tuneCNN')
    parser.add_argument('--learn_rate', type=float, default=0.001,
                        help='')
    parser.add_argument('--max_steps', type=int, default=50000,
                        help='')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.7,
                        help='for rnn cell')
    parser.add_argument('--adam_beta1', type=float, default=0.7,
                        help='beta1 for adam optimizer')
    parser.add_argument('--adam_epsilon', type=float, default=1.0e-8,
                        help='epsilon for adam optimizer')

    parser.add_argument('--train_img_model', type=int, default=0,
                        help='Fine-tune image model or not (0 as False, 1 as True)')
    parser.add_argument('--load_model_dir', type=str, default='',
                        help='path to the pre-trained whole model')
    parser.add_argument('--load_model_name', type=str, default='',
                        help='model name (model-%steps) of the pre-trained whole model')
    parser.add_argument('--experiment_path', type=str, default='result/speaker/temp',
                        help='where to save the training result')
    # Image feature
    parser.add_argument('--img_model', type=str, default='vgg_16',
                        help='alexnet, inception_v3, or vgg_16')
    parser.add_argument('--img_input_mode', type=str, default='per_state',
                        help='initial or per_state')
    parser.add_argument('--img_end_layers', type=lambda s: [int(n) for n in s.split('x')], default='1024x512x-1',
                        help="Specify the size of fc layers connected by 'x'. " +
                             "-1 means concat two features. Must contain -1")
    # Sentence feature
    parser.add_argument('--word_count_thresh', type=int, default=5,
                        help='1 ~ 5')
    parser.add_argument('--max_text_length', type=int, default=17,
                        help='Max text length, including _STA sent1 _VS sent2 _EOS')
    parser.add_argument('--word_embed_type', type=str, default='one-hot',
                        help='word2vec, one-hot or char for character-based model')
    parser.add_argument('--word_embed_dim', type=int, default=512,
                        help="if mode is 'initial', need to make sure it equals to last img_end_layer")
    # RNN feature
    parser.add_argument('--rnn_cell', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--rnn_num_units', type=int, default=2048,
                        help='number os hidden units, also the hidden state size')
    parser.add_argument('--rnn_depth', type=int, default=1,
                        help='Number of cells in RNN')

    args = parser.parse_args()
    if args.load_model_dir != '':
        # Overwrite necessary args from pre-trained model
        with open(os.path.join(args.load_model_dir, 'config.json')) as data_file:
            arg_dict = json.load(data_file)
            args.speaker_mode = arg_dict['speaker_mode']
            args.img_model = arg_dict['img_model']
            args.img_input_mode = arg_dict['img_input_mode']
            args.img_end_layers = arg_dict['img_end_layers']
            args.word_count_thresh = arg_dict['word_count_thresh']
            args.word_embed_type = arg_dict['word_embed_type']
            args.word_embed_dim = arg_dict['word_embed_dim']
            args.rnn_cell = arg_dict['rnn_cell']
            args.rnn_num_units = arg_dict['rnn_num_units']
            args.rnn_depth = arg_dict['rnn_depth']

    args.vocab_file = 'vocabulary/word_vocab_%s_%d.npy' % (args.train_set_name, args.word_count_thresh)
    args.annFile_json_train = os.path.join(data_dir, 'visdiff_%s.json' % args.train_set_name)

    if args.load_model_dir != '' and not args.train_img_model:
        print '***** WARNING *****\n Choose to load pre-trained model, but not to train the CNN! '

    if not args.train_img_model:
        cnn_tag = 'fixCNN'
    else:
        cnn_tag = 'tuneCNN_%s' % args.load_model_name

    # CHANGE OUTPUT DIR HERE!
#    args.experiment_path = 'result/speaker/%s_%s_len_%d__%s' % (args.speaker_mode, cnn_tag, args.max_text_length,
#                                                                time.strftime('%Y-%m-%d-%H'))

    train(args)


def train(args):
    # save output to file
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
    log_f = open(args.experiment_path + '/output.log', 'w')
    print "save to:" + args.experiment_path

    # print and save configuration
    print '=============='
    print 'Configurations:'
    for arg in vars(args):
        print arg + ":" + str(getattr(args, arg))
    print '=============='
    config_f = open(os.path.join(args.experiment_path, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()

    # Load dataset
    train_dataset = VisDiff(args.annFile_json_train).dataset['annotations']

    # Load imgs_dict
    if not args.train_img_model:
        imgs_dict_fpath = 'img_feat/img_feat_dict/%s_dict.json' % args.train_set_name
        with open(imgs_dict_fpath) as imgs_dict_file:
            imgs_dict = json.load(imgs_dict_file)
    else:
        imgs_dict = []

    # get vocab and save word2vec embedding
    if args.word_embed_type == 'char':
        vocabulary = data_feeder.build_vocabulary(args.annFile_json_train, args.word_count_thresh, args.word_embed_type)
    else:
        vocabulary = np.load(args.vocab_file).item()

    # if args.word_embed_type == 'word2vec':
    #     word2vec_npy_path = "../models/word2vec_concise_"+str(args.word_count_thresh)+'.npy'

    if args.img_model == 'alexnet' or args.img_model == 'vgg_16':
        img_w = img_h = 224
    elif args.img_model == 'inception_v3':
        img_w = img_h = 299
    else:
        raise Exception("model type not supported: {}".format(args.img_model))

    # build data_feeder
    data_train = data_feeder.DataFeeder(train_dataset, vocabulary=vocabulary, train_img_model=args.train_img_model,
                                        img_dir=img_dir, img_dict=imgs_dict, feed_mode=args.speaker_mode,
                                        rand_flip=True, word_embed_type=args.word_embed_type,  # shuffle=False,
                                        max_length=args.max_text_length, img_h=img_h, img_w=img_w)  # , augment=False)

    g = tf.Graph()
    with g.as_default() as graph:
        # Create a session for running Ops on the Graph
        sess = tf.Session(graph=graph)

        # setup the network
        vocab_size = len(vocabulary)
        print 'vocab_size: %d' % vocab_size
        if args.speaker_mode == 'DS':
            CaptionNet = pair_net.CaptionNet
        else:  # args.speaker_mode == 'S'
            CaptionNet = single_net.CaptionNetSingle
        my_net = CaptionNet(args, vocab_size=vocab_size, mode='train', img_h=img_h, img_w=img_w)
        my_net.build()
        print 'Successfully built the CaptionNet graph!'

        # print '#All variables: %d' % np.size(tf.global_variables())
        # for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='img_end_fc'):
        #     print v.name
        # print '----------------'

        # Setup summary
        summary_dir = args.experiment_path + '/summaries'
        if tf.gfile.Exists(summary_dir):
            tf.gfile.DeleteRecursively(summary_dir)
        tf.gfile.MakeDirs(summary_dir)
        train_writer = tf.summary.FileWriter(summary_dir, graph)

        # img_model_scope
        if args.img_model == 'inception_v3':
            img_model_scope = 'InceptionV3'
        elif args.img_model == 'vgg_16':
            img_model_scope = 'vgg_16'
        else:
            img_model_scope = 'Alexnet'

        # Initialize or reload variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # Load pre-trained img model weights
        if args.train_img_model and args.img_model != 'alexnet':
            checkpoint_file = 'models/checkpoints/'+args.img_model+'.ckpt'
            img_model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=img_model_scope)
            # print '#img model variables: %d' % np.size(img_model_variables)
            # for v in img_model_variables:
            #     print v.name
            # print '----------------'
            saver = tf.train.Saver(img_model_variables)
            saver.restore(sess, checkpoint_file)

        # Load previous checkpoint with fixed CNN feature
        if args.load_model_dir:
            model_fname = os.path.join(args.load_model_dir, args.load_model_name)
            all_variables = {v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
            restore_variables = {v for v in all_variables if
                                 v not in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=img_model_scope) and
                                 v not in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='optimizer')}
            # print '#pre-trained variables: %d' % np.size(restore_variables)
            # for v in restore_variables:
            #     print v.name
            # print '----------------'
            saver = tf.train.Saver(restore_variables)
            saver.restore(sess, model_fname)

        # start training
        new_saver = tf.train.Saver(max_to_keep=0)
        for step in range(1, args.max_steps+1):
            # training
            train_batch = data_train.get_batch(args.batch_size)
            if args.speaker_mode == 'DS':
                train_dict = {
                    my_net.img1: train_batch['img1'],
                    my_net.img2: train_batch['img2'],
                    my_net.text: train_batch['encode_text'],
                    my_net.target_seq: train_batch['target_text'],
                    my_net.text_mask: train_batch['text_mask']}
            else:  # args.speaker_mode == 'S'
                train_dict = {
                    my_net.img1: train_batch['img'],
                    my_net.text: train_batch['encode_sent'],
                    my_net.target_seq: train_batch['target_sent'],
                    my_net.text_mask: train_batch['sent_mask']}

            summary, _, batch_loss = sess.run([my_net.merged_summary, my_net.train, my_net.batch_loss],
                                              feed_dict=train_dict)

            train_writer.add_summary(summary, step)
            log_str = 'epoch %d - step %d: batch_loss %.3f' % (data_train.epoch_count, step, batch_loss)
            print log_str
            log_f.write(log_str + '\n')

            if np.isnan(batch_loss):
                print('Model diverged with loss = NaN')
                quit()

            # save the model, do evaluation
            if step % 5000 == 0 or step == args.max_steps:
                save_dir = os.path.join(args.experiment_path, 'model')
                new_saver.save(sess, save_dir, global_step=step)

                # score_record_list = inference(args, args.experiment_path, 'model-%d' % step, output_num=3,
                #                               out_to_html=1, return_scores=1, case_num=5)
                # score_mean = {}
                # out_num = len(score_record_list)
                # for score_record in score_record_list:
                #     for method in score_record.iterkeys():
                #         if score_mean.get(method) is None:
                #             score_mean[method] = score_record[method] / out_num
                #         else:
                #             score_mean[method] += score_record[method] / out_num
                #
                # summary = tf.Summary()
                # for method in score_mean.iterkeys():
                #     summary.value.add(tag=method, simple_value=score_mean[method])
                #     print 'EVAL: %s %f' % (method, score_mean[method])
                # train_writer.add_summary(summary, step)

        log_f.close()


if __name__ == "__main__":
    main()
