import tensorflow as tf
import numpy as np
import time, os, json, argparse
from pyVisDifftools.visdiff import VisDiff
import data_loader as data
import network.listener_net as net
#from utils.prepare_embedding import save_concise_emb
from utils.train_listener_args_parser import train_args_parser
import progressbar # pip install progressbar if not installed

def main():
    parser = train_args_parser()
    train_args = parser.parse_args()
    if train_args.input_path != '':
        load_arguments(parser, os.path.join(train_args.input_path, 'config.json'))
    train_args.vocab_file = 'vocabulary/word_vocab_train_%d.npy' % train_args.word_count_thresh
    train(train_args)

def load_arguments(parser, json_fpath):
    with open(json_fpath) as data_file:
        arg_dict = json.load(data_file)
    for (key, value) in arg_dict.items():
        parser.add_argument('--' + key, default=value)

def train(args):
    # Set dataset path
    annFile_json_train = 'dataset/visdiff_train.json'

    # Load img_feat_dict, which maps the image name to the id in numpy file
    # we save image feature extraced from pre-trained imagenet to accelerate training
    # for the first training step (fix image feature in the listener model)
    all_imgs_dict = []
    if not args.train_img_model:
        all_imgs_dict = json.load(open('img_feat/img_feat_dict/'+str(args.dataset)+'_dict.json'))

    # Load/Build vocabulary
    if args.word_embed_type == 'char':
        vocabulary = data.build_vocabulary(annFile_json_train, args.word_count_thresh, args.word_embed_type)
    else:
        if tf.gfile.Exists(args.vocab_file):
            vocabulary = np.load(args.vocab_file).item()
        else:
            vocabulary = data.build_vocabulary(annFile_json_train, args.word_count_thresh, args.word_embed_type)

    # Load word2vec embeddings (normally we use one-hot encoding instead)
    if args.word_embed_type == 'word2vec':
        word2vec_npy_path = "word2vec/word2vec_concise_"+str(args.word_count_thresh)+'.npy'
        if not tf.gfile.Exists(word2vec_npy_path):
            save_concise_emb(vocabulary, word2vec_npy_path)

    # Build data_loader
    if args.img_model=='vgg_16':
        img_w = img_h = 224
    elif args.img_model=='inception_v3':
        img_w = img_h = 299
    else:
        raise Exception("model type not supported: {}".format(args.img_model))

    if args.pairwise:
        feed_mode = 'DL'
    else:
        feed_mode = 'SL'

    if args.mode == 'train':
        train_dataset = VisDiff(annFile_json_train).dataset['annotations']
        data_train = data.DataFeeder(train_dataset, vocabulary=vocabulary, train_img_model=args.train_img_model,
                                     img_dict=all_imgs_dict, feed_mode=feed_mode, rand_neg=args.ran_neg_sample,
                                     max_length=args.max_sent_length, img_w=img_w, img_h=img_h,
                                     word_embed_type=args.word_embed_type)
    elif args.mode == 'eval':
        annFile_json_val = 'dataset/visdiff_%s.json'%(args.dataset)
        val_dataset = VisDiff(annFile_json_val).dataset['annotations']
        data_val = data.DataFeeder(val_dataset, vocabulary=vocabulary, train_img_model=args.train_img_model,
                                   img_dict=all_imgs_dict, feed_mode=feed_mode, rand_neg=False,
                                   max_length=args.max_sent_length, img_w=img_w, img_h=img_h,
                                   word_embed_type=args.word_embed_type, trim_last_batch=True, shuffle=False)

    # Set result path
    EXPERIMENT_PATH = args.log_dir
    if not os.path.exists(EXPERIMENT_PATH):
        os.makedirs(EXPERIMENT_PATH)
    print "save result to:" + EXPERIMENT_PATH

    log_dir = EXPERIMENT_PATH+'/summaries'
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    print "save summary to:" + log_dir

    # Save configurations
    config_f = open(os.path.join(EXPERIMENT_PATH, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()

    # Training
    with tf.Graph().as_default() as graph:
        # Create a session for running Ops on the Graph
        sess = tf.Session()

        # Setup the network
        my_net = net.listener_net(args, vocab_length=len(vocabulary),
            graph=sess.graph, log_dir=log_dir, img_w=img_w, img_h=img_h, dataset=args.dataset)
        my_net.build()

        # Initialize the variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # Load pre-trained model weights
        if args.img_model == 'inception_v3':
            scope_name = 'InceptionV3'
        elif args.img_model == 'vgg_16':
            scope_name = 'vgg_16'
        else:
            print ('image model is not found')

        # Load cnn model pre-trained on imagenet
        if args.train_img_model and args.mode == 'train' :
            checkpoint_file = 'models/checkpoints/'+args.img_model+'.ckpt'
            img_model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
            saver = tf.train.Saver(img_model_variables)
            saver.restore(sess, checkpoint_file)

        # Load previous checkpoint with fixed CNN feature
        if args.load_model_path != 'none':
            model_fname = os.path.join(EXPERIMENT_PATH, args.load_model_path)
            all_variables = {v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
            other_variables = {v for v in all_variables if v not in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='optimizer')
                                and v not in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='embeddings')}
            if not args.train_img_model:# then don't need to load cnn model, just use extracted feature (saved in numpy)
                other_variables = {v for v in other_variables if v not in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)}
            saver = tf.train.Saver(other_variables)
            saver.restore(sess, model_fname)

        if args.mode == 'train':
            # Start training
            saver = tf.train.Saver(max_to_keep=1)
            for step in range(args.max_steps):
                train_batch = data_train.get_batch(args.batch_size)
                if args.pairwise:
                    train_dict = {
                        my_net.img1: train_batch['img1'],
                        my_net.img2: train_batch['img2'],
                        my_net.text1: train_batch['encode_text1'],
                        my_net.text2: train_batch['encode_text2'],
                        my_net.text_len: train_batch['text_len'],
                        my_net.is_training_ph: True}
                else:
                    train_dict = {
                        my_net.img1: train_batch['img1'],
                        my_net.img2: train_batch['img2'],
                        my_net.text1: train_batch['encode_sent1'],
                        my_net.text2: train_batch['encode_sent2'],
                        my_net.s_len1: train_batch['sent_len1'],
                        my_net.s_len2: train_batch['sent_len2'],
                        my_net.is_training_ph: True}

                summary, _, loss_train, acc_img2txt, acc_txt2img = sess.run([my_net.merged, my_net.train, my_net.loss,
                                                                 my_net.acc_img2txt, my_net.acc_txt2img], feed_dict=train_dict)
                my_net.train_writer.add_summary(summary, step+1)

                if (step+1) % 10 == 0:
                    log_str = 'epoch %d, step %d: train_loss %.3f; train_acc_img2txt %.3f; train_acc_txt2img %.3f' \
                              % (data_train.epoch_count+1, step+1, loss_train, acc_img2txt, acc_txt2img)
                    print log_str

                if np.isnan(loss_train):
                    print('Model diverged with loss = NaN')
                    quit()

                # Save the model, for initialize and re-train in the future
                if (step+1) % 500 == 0 or (step+1) == args.max_steps:
                    if args.train_img_model:
                        save_dir = EXPERIMENT_PATH + '/model-finetune'
                    else:
                        save_dir = EXPERIMENT_PATH + '/model-fixed'
                    saver.save(sess, save_dir, global_step=step+1)

        elif args.mode == 'eval':
            # Start evaluation
            log_f = open(EXPERIMENT_PATH+'/'+args.mode+'.log', 'w')
            start_iter_count = data_val.epoch_count
            loss_val_all = []
            acc_txt2img_val_all = []
            acc_txt2img_w12_val_all = []
            progress_count = 0
            bar = progressbar.ProgressBar(maxval=len(data_val.annotations)*2, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            while start_iter_count == data_val.epoch_count:
                val_batch = data_val.get_batch(args.batch_size)
                if args.pairwise:
                    val_dict = {
                        my_net.img1: val_batch['img1'],
                        my_net.img2: val_batch['img2'],
                        my_net.text1: val_batch['encode_text1'],
                        my_net.text2: val_batch['encode_text2'],
                        my_net.text_len: val_batch['text_len'],
                        my_net.is_training_ph: False}
                    batch_count = len(val_batch['text_len'])*2
                else:
                    val_dict = {
                        my_net.img1: val_batch['img1'],
                        my_net.img2: val_batch['img2'],
                        my_net.text1: val_batch['encode_sent1'],
                        my_net.text2: val_batch['encode_sent2'],
                        my_net.s_len1: val_batch['sent_len1'],
                        my_net.s_len2: val_batch['sent_len2'],
                        my_net.is_training_ph: False}
                    batch_count = len(val_batch['sent_len1'])*2

                progress_count += batch_count
                bar.update(progress_count)
                loss_val, acc_txt2img, acc_txt2img_w12 = sess.run(
                    [my_net.loss, my_net.acc_txt2img, my_net.acc_txt2img_w12], feed_dict=val_dict)
                loss_val_all.append(loss_val/batch_count)
                acc_txt2img_val_all.append(acc_txt2img*batch_count)
                acc_txt2img_w12_val_all.append(acc_txt2img_w12*batch_count)

            bar.finish()
            loss_val_all = np.sum(loss_val_all)*args.batch_size/(len(data_val.annotations)*2)
            acc_txt2img_val_all = np.sum(acc_txt2img_val_all)/(len(data_val.annotations)*2)
            acc_txt2img_w12_val_all = np.sum(acc_txt2img_w12_val_all)/(len(data_val.annotations)*2)
            if feed_mode == 'SL':
                log_str = 'Accuracy on %s set (using model %s): acc_SL %.3f; acc_2xSL %.3f ' % (args.dataset,
                          args.load_model_path, acc_txt2img_val_all, acc_txt2img_w12_val_all)
            elif feed_mode == 'DL':
                log_str = 'Accuracy on %s set (using model %s): acc_DL %.3f;' % (args.dataset,
                          args.load_model_path, acc_txt2img_val_all)
            print log_str
            log_f.write(log_str + '\n')
            log_f.close()

if __name__ == '__main__':
    main()
