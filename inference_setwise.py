""" Generate descriptions that differentiate two sets of images
"""

import os
import argparse
import operator
import random
import numpy as np
from scipy.misc import imread
import tensorflow as tf
import scipy.io as sio

from utils import caption_generator
from inference_pairwise import InferenceWrapperPair, InferenceWrapperSingle, captions_id2word, load_arguments

oid_anno_path = 'dataset/oid/anno.mat'
fgvc_variant_dir = 'dataset/fgvc-aircraft-2013b/data'
img_dir = 'dataset/images'
html_dir = 'result/inference_set'
html2img_dir = '../../dataset/images'

speaker_path = 'result/speaker/DS_tuneCNN_model-40000__2017-03-07-19/'


def main(_):
    exp_num = 40
    sample_num = 10
    top_n = 20

    model_dict = load_fgvc_model_dict()
    # model_list = [model for model, img_list in model_dict.iteritems() if len(img_list) > img_num]
    txt_f = open(os.path.join(html_dir, 'inference_set_top20.txt'), 'w')
    html_f = open(os.path.join(html_dir, 'inference_set_top20.html'), 'w')
    exp = 0
    while exp < exp_num:
        model_sample = random.sample(model_dict.keys(), 2)
        if exp == 0:
            model_sample = ['747-400', 'ATR-42']
        if model_sample[0][:3] == model_sample[1][:3]:
            continue
        exp += 1
        img1_list = []
        img2_list = []
        # for i in range(sample_num):
        img1_list += random.sample(model_dict[model_sample[0]], sample_num)
        img2_list += random.sample(model_dict[model_sample[1]], sample_num)

        pos_sc, neg_sc = k2k_diff(img1_list, img2_list, top_n, rand=0)
        print exp
        print model_sample
        print img1_list
        print img2_list
        print pos_sc
        print neg_sc
        print '=====================\n'
        txt_f.write('%d\n%s\n%s\n%s\n%s\n%s\n\n'
                    % (exp, str(model_sample), str(img1_list), str(img2_list), str(pos_sc), str(neg_sc)))
        output_as_html(html_f, img1_list, img2_list, pos_sc + neg_sc)

    txt_f.close()
    html_f.close()


def output_as_html(html_f, img1_list, img2_list, sents_count):
    html_f.write('<table align=\'center\'>\n')
    msg = ''
    for img1 in img1_list:
        msg += '<img src=\'%s\' height=100>\n' % (os.path.join(html2img_dir, img1))
    html_f.write('<tr><td>\n' + msg + '</td></tr>\n')

    msg = ''
    for img2 in img2_list:
        msg += '<img src=\'%s\' height=100>' % (os.path.join(html2img_dir, img2))
    html_f.write('<tr><td>' + msg + '</td></tr>\n')

    for sent, count in sents_count:
        msg = '[%d] %s' % (count, sent)
        html_f.write('<tr><td>' + msg + '</td></tr>\n')

    html_f.write('<tr> <td colspan="4" bgcolor="#009966" height="3">&nbsp;</td> </tr>\n')
    html_f.write('</table>\n')


def load_OID_model_dict():
    # load OID annotation
    anno = sio.loadmat(oid_anno_path)

    # oid_anno['airline'] = anno['anno'][0,0]['aeroplane'][0,0]['attribute'][0,0]['airline']  # (1,7426)
    model_list = anno['anno'][0,0]['aeroplane'][0,0]['attribute'][0,0]['model'][0]  # (1,7426)
    parentID_list = anno['anno'][0,0]['aeroplane'][0,0]['parentId'][0]  # in range(1,7413) in shape(1,7426)
    img_name_list = anno['anno'][0,0]['image'][0,0]['name'][0,:].tolist()

    model_dict = dict()
    for i in range(len(model_list)):
        model = model_list[i]
        pid = parentID_list[i] - 1
        img_name = img_name_list[pid]
        if model_dict.get(model) is None:
            model_dict[model] = list(img_name)
        else:
            model_dict[model].append(img_name[0])
    return model_dict


def load_fgvc_model_dict(set_split='trainval'):
    model_dict = {}
    file_path = os.path.join(fgvc_variant_path, 'images_variant_%s.txt' % set_split)
    sentences = open(file_path).read().strip().split("\n")
    for i in range(len(sentences)):
        img_name, model = sentences[i].split(" ", 1)
        if model_dict.get(model) is None:
            model_dict[model] = [img_name + '.jpg']
        else:
            model_dict[model].append(img_name + '.jpg')
    return model_dict


def k2k_diff(img1_ids, img2_ids, top_n, rand=1):
    pair_cases = []
    if rand == 0:
        for img1_id in img1_ids:
            for img2_id in img2_ids:
                case = dict()
                case['img1_id'] = img1_id
                case['img2_id'] = img2_id
                pair_cases.append(case)
    else:
        for i in range(len(img1_ids)):
            case = dict()
            case['img1_id'] = img1_ids[i]
            case['img2_id'] = img2_ids[i]
            pair_cases.append(case)

    parser = argparse.ArgumentParser()
    load_arguments(parser, os.path.join(speaker_path, 'config.json'))
    args = parser.parse_args()

    sent1_list, sent2_list = inference_pairs(pair_cases, args, os.path.join(speaker_path, 'model-40000'))
    return rank_by_freq(sent1_list, sent2_list, top_n, len(img1_ids))


def inference_pairs(cases, speaker_args, speaker_model_path, output_num=10, beam_size=10):
    # Get the vocabulary.
    vocab = np.load(speaker_args.vocab_file).item()
    speaker_args.vocab_size = len(vocab)

    # Build reverse vocabulary
    word_list = [''] * len(vocab)
    for w, case_i in vocab.iteritems():
        word_list[case_i] = w

    # Set img size
    img_w = img_h = 224
    speaker_args.img_h = img_h
    speaker_args.img_w = img_w

    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        if speaker_args.speaker_mode == 'S':
            infer_model = InferenceWrapperSingle(speaker_args.img_input_mode)
        else:  # mode 'DS'
            infer_model = InferenceWrapperPair(speaker_args.img_input_mode)

        # print and save configuration
        speaker_args.batch_size = 1
        # print '=============='
        # print 'Configurations:'
        # for arg in vars(speaker_args):
        #     print arg + ":" + str(getattr(speaker_args, arg))
        # print '=============='

        restore_fn = infer_model.build_graph_from_config(speaker_args, speaker_model_path)

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
        generator = caption_generator.CaptionGenerator(infer_model, vocab, beam_size=beam_size)
        sent1_list = []
        sent2_list = []
        for case_i, case in enumerate(cases):
            img1_id = case['img1_id']
            img2_id = case['img2_id']
            print '\t', case_i, img1_id, img2_id
            img1 = imread(os.path.join(img_dir, str(img1_id)), mode='RGB')
            img2 = imread(os.path.join(img_dir, str(img2_id)), mode='RGB')

            gen_sen_id = generator.beam_search(sess, img1, img2)
            gen_sen_w, gen_probs = captions_id2word(gen_sen_id[0: output_num], word_list, spliter=' ')
            print '\t', gen_sen_w, '\n-------------------\n'
            for text in gen_sen_w:
                sp = text.split(' _VS ')
                if len(sp) >= 2:
                    sent1 = sp[0]
                    sent2 = sp[1]
                else:
                    sent1 = sp[0]
                    sent2 = ''
                sent1_list.append(sent1.strip())
                sent2_list.append(sent2.strip())

    return sent1_list, sent2_list


def rank_by_freq(pos_list, neg_list, top_n, img_count):
    counter_dict = {}

    for i in range(len(pos_list)):
        s = pos_list[i]
        if counter_dict.get(s) is None:
            counter_dict[s] = np.zeros((2, img_count))
        counter_dict[s][0, (i / 10) / img_count] += 1
    for i in range(len(neg_list)):
        s = neg_list[i]
        if counter_dict.get(s) is None:
            counter_dict[s] = np.zeros((2, img_count))
        counter_dict[s][1, (i / 10) % img_count] += 1
    np.save('counter_dict.npy', counter_dict)
    score_dict = {}
    for s, c in counter_dict.iteritems():
        if '_UNK' in s:
            continue
        score = sum(c[0] > 0) - sum(c[1] > 0) + (sum(c[0]) - sum(c[1])) / (10.0 * img_count * img_count)
        score_dict[s] = score
    sorted_s = sorted(score_dict.items(), key=operator.itemgetter(1))
    pos_s = sorted_s[-top_n:]
    pos_s.reverse()
    neg_s = sorted_s[:top_n]
    return pos_s, neg_s


if __name__ == "__main__":
    tf.app.run()
