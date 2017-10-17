""" Given a pair of images, generate 'S1 vs. S2' using a trained decerning speaker model.
    Given one image, genersate its description using a trained single speaker model.
    Output: 
        html file that collects image pairs and captions
        save the dataset with captions and evaluation metrics scores
        
"""

import os
import sys
import math
import json
import time
import argparse
import numpy as np
import tensorflow as tf

from utils.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from utils.pycocoevalcap.bleu.bleu import Bleu
from utils.pycocoevalcap.meteor.meteor import Meteor
from utils.pycocoevalcap.rouge.rouge import Rouge
from utils.pycocoevalcap.cider.cider import Cider

from utils import caption_generator, inference_wrapper_base
from network.speaker_net_discerning import CaptionNet as CaptionNetPair
from network.speaker_net_simple import CaptionNetSingle
from pyVisDifftools.visdiff import VisDiff
import data_loader as data_feeder

html_dir = 'result/inference'
data_dir = 'dataset'
html2img_dir = '../../dataset/images'
# html2img_dir = os.path.join(os.path.abspath(data_dir), 'images')


class InferenceWrapperSingle(inference_wrapper_base.InferenceWrapperBase):
    """Model wrapper class for performing inference with CaptionNet."""

    def __init__(self, img_input_mode):
        super(InferenceWrapperSingle, self).__init__()
        self.img_input_mode = img_input_mode
        self.img = None
        self.model = None

    def build_model(self, args):
        model = CaptionNetSingle(args, mode="inference", vocab_size=args.vocab_size, img_h=args.img_h, img_w=args.img_w)
        model.build()
        print 'vocab_size: %d' % model.vocab_size
        self.model = model
        return model

    def img_initial_state(self, sess, image1, image2=None):
        self.img = np.expand_dims(image1, 0)
        # print '*** img ***'
        # print image1
        initial_state = sess.run(fetches=["rnn_1/initial_state:0"],
                                 feed_dict={self.model.img1: self.img})
        return initial_state

    def inference_step(self, sess, input_feed, state_feed):
        sp = np.shape(state_feed)
        if np.size(sp) == 3 and sp[1] == 1:
            state_feed = state_feed[:, 0, :]

        if self.img_input_mode is 'initial':
            softmax_output, state_output = sess.run(fetches=["softmax:0", "rnn_1/state:0"],
                                                    feed_dict={"text_seq:0": input_feed,
                                                               "rnn_1/state_feed:0": state_feed})
        else:  # self.img_input_mode is 'per_state':
            len = np.shape(input_feed)[0]
            new_shape = np.ones(np.shape(np.shape(self.img)), dtype=np.int8)
            new_shape[0] = len
            img = np.tile(self.img, new_shape)
            softmax_output, state_output = sess.run(fetches=["softmax:0", "rnn_1/state:0"],
                                                    feed_dict={"image1:0": img,
                                                               "text_seq:0": input_feed,
                                                               "rnn_1/state_feed:0": state_feed})
        return softmax_output, state_output, None


class InferenceWrapperPair(inference_wrapper_base.InferenceWrapperBase):
    """Model wrapper class for performing inference with CaptionNet."""
    def __init__(self, img_input_mode):
        super(InferenceWrapperPair, self).__init__()
        self.img_input_mode = img_input_mode
        self.img1 = None
        self.img2 = None
        self.model = None

    def build_model(self, args):
        model = CaptionNetPair(args, mode="inference", vocab_size=args.vocab_size, img_h=args.img_h, img_w=args.img_w)
        model.build()
        print 'vocab_size: %d' % model.vocab_size
        self.model = model
        return model

    def img_initial_state(self, sess, image1, image2):
        self.img1 = np.expand_dims(image1, 0)
        self.img2 = np.expand_dims(image2, 0)
        initial_state = sess.run(fetches=["rnn_1/initial_state:0"],
                                 feed_dict={self.model.img1: self.img1, self.model.img2: self.img2})
        return initial_state

    def inference_step(self, sess, input_feed, state_feed):
        sp = np.shape(state_feed)
        if np.size(sp) == 3 and sp[1] == 1:
            state_feed = state_feed[:, 0, :]

        if self.img_input_mode is 'initial':
            softmax_output, state_output = sess.run(fetches=["softmax:0", "rnn_1/state:0"],
                                                    feed_dict={"text_seq:0": input_feed,
                                                               "rnn_1/state_feed:0": state_feed})
        else:  # self.img_input_mode is 'per_state':
            len = np.shape(input_feed)[0]
            new_shape = np.ones(np.shape(np.shape(self.img1)), dtype=np.int8)
            new_shape[0] = len
            img1 = np.tile(self.img1, new_shape)
            img2 = np.tile(self.img2, new_shape)
            softmax_output, state_output = sess.run(fetches=["softmax:0", "rnn_1/state:0"],
                                                    feed_dict={"image1:0": img1,
                                                               "image2:0": img2,
                                                               "text_seq:0": input_feed,
                                                               "rnn_1/state_feed:0": state_feed})
        return softmax_output, state_output, None


def captions_id2word(captions, word_list, spliter=' '):
    sentences = []
    probs = []
    for i, caption in enumerate(captions):
        sent = ''
        for w_id in caption.sentence[1:-1]:
            sent += word_list[w_id] + spliter
        sentences.append(sent)
        probs.append(math.exp(caption.logprob))
    return sentences, probs


def _infer_wirte_as_txt(result_f, img_id1, img_id2, true_s1, true_s2, gen_sen_w, gen_probs):
    msg = "\n*** image %s vs %s ***" % (img_id1, img_id2)
    result_f.write(msg + '\n')
    msg = "Ground Truth:"
    result_f.write(msg + '\n')
    for i in range(5):
        msg = '  %s _VS %s' % (true_s1[i], true_s2[i])
        result_f.write(msg + '\n')
    msg = "Generated:"
    result_f.write(msg + '\n')
    for i in range(len(gen_probs)):
        msg = "  %d) %s \t (p=%f)" % (i+1, gen_sen_w[i], gen_probs[i])
        result_f.write(msg + '\n')


def _infer_wirte_as_html(result_f, img_id1, img_id2, true_s1, true_s2, gen_sen_w, gen_probs, img_dir=html2img_dir):
    msg = "*** image %s vs %s ***" % (img_id1, img_id2)
    result_f.write('<tr align=\'center\'><td>' + msg + '</td></tr>\n')
    msg = '<img src=\'%s.jpg\' height=150><img src=\'%s.jpg\' height=150>' \
          % (os.path.join(img_dir, img_id1), os.path.join(img_dir, img_id2))
    result_f.write('<tr align=\'center\'><td>' + msg + '</td></tr>\n')
    msg = "Ground Truth:"
    result_f.write('<tr align=\'center\'><td>' + msg + '</td></tr>\n')
    for i in range(5):
        msg = '  %s _VS %s' % (true_s1[i], true_s2[i])
        result_f.write('<tr align=\'center\'><td>' + msg + '</td></tr>\n')
    msg = "Generated:"
    result_f.write('<tr align=\'center\'><td>' + msg + '</td></tr>\n')
    for i in range(len(gen_probs)):
        msg = "  %d) %s \t (p=%f)" % (i+1, gen_sen_w[i], gen_probs[i])
        result_f.write('<tr align=\'center\'><td>' + msg + '</td></tr>\n')
    result_f.write('<tr> <td colspan="4" bgcolor="#009966" height="3">&nbsp;</td> </tr>\n')


def inference(args, input_path, model_step, dataset_name='val', case_num=0, output_num=10, beam_size=10,
              out_to_html=50, eval_scores=1, save_dataset=False):

    if save_dataset:
        case_num = 0

    model_path = os.path.join(input_path, model_step)
    data_json_file = os.path.join(data_dir, 'visdiff_%s.json' % dataset_name)
    args.batch_size = 1  # inference for one image at a time

    # Load dataset
    dataset = VisDiff(data_json_file).dataset['annotations']

    # Load imgs_dict
    if not args.train_img_model:
        imgs_dict_fpath = '../imgs_dict/%s_dict.json' % dataset_name
        with open(imgs_dict_fpath) as imgs_dict_file:
            imgs_dict = json.load(imgs_dict_file)
    else:
        imgs_dict = []

    # Get the vocabulary.
    if args.word_embed_type == 'char':
        vocab = data_feeder.build_vocabulary('%s/visdiff_trainval.json'
                                             % data_dir, args.word_count_thresh, args.word_embed_type)
    else:
        vocab = np.load(args.vocab_file).item()
    args.vocab_size = len(vocab)

    # Build reverse vocabulary
    word_list = [''] * len(vocab)
    for w, case_i in vocab.iteritems():
        word_list[case_i] = w

    # Set img size
    if args.img_model == 'alexnet' or args.img_model == 'vgg_16':
        img_w = img_h = 224
    elif args.img_model == 'inception_v3':
        img_w = img_h = 299
    else:
        raise Exception("model type not supported: {}".format(args.img_model))
    args.img_h = img_h
    args.img_w = img_w

    # build data_feeder
    my_feeder = data_feeder.DataFeeder(dataset, vocabulary=vocab, train_img_model=args.train_img_model,
                                       img_dir='../../../dataset/images_224/', img_dict=imgs_dict,
                                       max_length=args.max_text_length, img_w=img_w, img_h=img_h)
    cases = dataset
    if case_num > 0:
        cases = dataset[0: case_num]

    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        if args.speaker_mode == 'S':
            infer_model = InferenceWrapperSingle(args.img_input_mode)
        else:  # mode 'DS'
            infer_model = InferenceWrapperPair(args.img_input_mode)
        restore_fn = infer_model.build_graph_from_config(args, model_path)

    # g.finalize()
    tf.reset_default_graph()
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    # with tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session(graph=g) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # Load the model from checkpoint.
        restore_fn(sess)
        g.finalize()
        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        if eval_scores:
            true_text_dict = {}
            gen_text_collect = []
            for sent_j in range(output_num):
                gen_text_collect.append({})

        generator = caption_generator.CaptionGenerator(infer_model, vocab, beam_size=beam_size)
        spliter = ' '
        if args.word_embed_type == 'char':
            spliter = ''

        if out_to_html:
            result_path = os.path.join(html_dir, 'infer_%s_%s_%d.html' % (dataset_name, model_step, beam_size))
            result_f = open(result_path, 'w')
            result_f.write('<table align=\'center\'>\n')

        for case_i, case in enumerate(cases):
            img_id1 = case['img1_id']
            img_id2 = case['img2_id']
            img1 = my_feeder.get_img(img_id1, args.img_w, args.img_h)
            img2 = my_feeder.get_img(img_id2, args.img_w, args.img_h)
            true_s1 = case['sentences1']
            true_s2 = case['sentences2']

            gen_sen_id = generator.beam_search(sess, img1, img2)
            gen_sen_w, gen_probs = captions_id2word(gen_sen_id[0: output_num], word_list, spliter)

            if out_to_html > 0 and case_i < out_to_html:
                _infer_wirte_as_html(result_f, img_id1, img_id2, true_s1, true_s2, gen_sen_w, gen_probs)

            if eval_scores:
                true_text_dict[case_i*10] = []
                true_text_dict[case_i*10+1] = []
                for idx in range(5):
                    # true_text_dict[case_i].append(true_s1[idx] + ' _VS ' + true_s2[idx])
                    true_text_dict[case_i*10].append(true_s1[idx])
                    true_text_dict[case_i*10 + 1].append(true_s2[idx])
                for sent_j in range(output_num):
                    ss = gen_sen_w[sent_j].split('_VS')
                    left_s = ss[0].strip()
                    if len(ss) > 1:
                        right_s = ss[1].strip()
                    else:
                        right_s = ''
                    gen_text_collect[sent_j][case_i*10] = [left_s]
                    gen_text_collect[sent_j][case_i*10+1] = [right_s]

            if save_dataset:
                case['gen_texts'] = gen_sen_w
                case['gen_probs'] = gen_probs
                if eval_scores:
                    case['scores'] = {}

            print '%s: %d / %d' % (time.strftime('[%H:%M:%S]'), case_i, len(cases))

        if out_to_html:
            result_f.write('</table>\n')

        if eval_scores:
            score_record_list, scores_record_list = _eval_infer(true_text_dict, gen_text_collect)

            if out_to_html:
                msg = '>> ' + '  '.join(score_record_list[0].iterkeys())
                result_f.write(msg + '<br/>\n')

            score_mean = {}
            out_num = len(score_record_list)
            for sent_j, score_record in enumerate(score_record_list):
                if out_to_html:
                    msg = '%d)\t' % sent_j + '\t'.join('%.3f' % sc for sc in score_record.itervalues())
                    result_f.write(msg + '<br/>\n')

                for method in score_record.iterkeys():
                    if sent_j == 0:
                        score_mean[method] = score_record[method] / out_num
                    else:
                        score_mean[method] += score_record[method] / out_num

            # save overall scores to json
            to_save = {'input_path': input_path, 'model_step': model_step, 'score_record_list': score_record_list,
                       'score_mean': score_mean}
            score_f = open(os.path.join(input_path, 'infer_scores_%s_%s_case%d_beam%d_sent%d.json'
                                        % (dataset_name, model_step, case_num, beam_size, output_num)), 'w')
            json.dump(to_save, score_f)
            score_f.close()

            if save_dataset:
                if eval_scores:
                    for case_i, case in enumerate(cases):
                        for method in score_record_list[0].iterkeys():
                            case['scores'][method] = [[0.0, 0.0]] * output_num
                            for sent_i in range(output_num):
                                scores = scores_record_list[sent_i][method]
                                case['scores'][method][sent_i] = [scores[case_i*2], scores[case_i*2 + 1]]
                dataset_f = open(os.path.join(input_path, 'infer_annotations_%s_%s_case%d_beam%d_sent%d.json'
                                              % (dataset_name, model_step, case_num, beam_size, output_num)), 'w')
                json.dump(cases, dataset_f)
                dataset_f.close()

            if out_to_html:
                msg = 'ave\t' + '\t'.join('%.3f' % sc for sc in score_mean.itervalues())
                result_f.write(msg + '<br/>\n')

        if out_to_html:
            result_f.close()


def _eval_infer(true_text_dict, gen_text_collect):
    score_record_list = []
    scores_record_list = []
    for i, gen_txt_dict in enumerate(gen_text_collect):
        score_record = {}
        scores_record = {}
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(true_text_dict)
        res = tokenizer.tokenize(gen_txt_dict)

        # print 'setting up scorers...'
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)

            if type(method) == list:
                for sc, ss, m in zip(score, scores, method):
                    score_record[m] = sc
                    scores_record[m] = ss
            else:
                score_record[method] = score
                scores_record[method] = scores

        score_record_list.append(score_record)
        scores_record_list.append(scores_record)
    return score_record_list, scores_record_list


def main(_):
    parser = argparse.ArgumentParser()

    parser.add_argument('--case_num', type=int, default=0,
                        help='Number of examples to run, <1 means whole dataset')
    parser.add_argument('--out_to_html', type=int, default=50,
                        help='Number of examples to be shown in html. 0 means no html file')
    parser.add_argument('--output_num', type=int, default=10,
                        help='Number of sentences for each example')
    parser.add_argument('--beam_size', type=int, default=10,
                        help='For beam search. Must guarantee beam_size >= output_num')

    parser.add_argument('--input_path', type=str,
                        default='result/speaker/DS_fixCNN__2017-03-06-23',
                        help='Directory with model and config.json inside')
    parser.add_argument('--model_step', type=str, default='model-50000',
                        help='The model to use inside the input_path')
    parser.add_argument('--dataset_name', type=str, default='val',
                        help='The dataset_name used for inference, train / val / test')

    parser.add_argument('--eval_scores', type=int, default=1,
                        help='Whether or not compute evaluation scores.')
    parser.add_argument('--save_dataset', type=int, default=1,
                        help='Whether or not save the dataset with inference and scores added.')

    args = parser.parse_args()
    load_arguments(parser, os.path.join(args.input_path, 'config.json'))
    args = parser.parse_args()
    args.train_set_name =args.dataset_name

    # print vars(args)
    inference(args=args, input_path=args.input_path, model_step=args.model_step,
              dataset_name=args.dataset_name, case_num=args.case_num, output_num=args.output_num,
              beam_size=args.beam_size, out_to_html=args.out_to_html, eval_scores=args.eval_scores,
              save_dataset=args.save_dataset)


def load_arguments(parser, json_fpath):
    with open(json_fpath) as data_file:
        arg_dict = json.load(data_file)
    for (key, value) in arg_dict.items():
        parser.add_argument('--' + key, default=value)


if __name__ == "__main__":
    tf.app.run()
