import tensorflow as tf
import numpy as np
import time, os, json
from scipy.misc import imread, imresize
import sys
import progressbar # pip install progressbar if not installed
from train_args_parser import train_args_parser
sys.path.insert(0, '../')
import network.get_feature_net as net
from pyVisDifftools.visdiff import VisDiff
import data_loader as data

def main():
    parser = train_args_parser()
    args = parser.parse_args()
    # Set dataset path
    annFile_json = '../dataset/visdiff_%s.json' % (args.dataset)

    # Load imgs_dict
    all_imgs = VisDiff(annFile_json).dataset['img_all_id']
    all_imgs_dict = dict(zip(all_imgs, range(len(all_imgs))))

    outFileName = '../img_feat/img_feat_dict/%s_dict.json' % (args.dataset)
    with open(outFileName, 'w') as outfile:
        json.dump(all_imgs_dict, outfile)
    
    # Build data_feeder
    if args.img_model=='alexnet' or args.img_model=='vgg_16':
        img_w = img_h = 224
        feat_dim = 4096
    elif args.img_model=='inception_v3':
        img_w = img_h = 299
        feat_dim = 2048
    else:
        raise Exception("model type not supported: {}".format(args.img_model))

    print args.img_model
    print img_w
    print feat_dim

    with tf.Graph().as_default():
  
        # Create a session for running Ops on the Graph
        sess = tf.Session()

        # Setup the network
        mode = 'eval'
        my_net = net.image_net(args, graph=sess.graph, img_w=img_w, img_h=img_h)
        my_net.build()

        # Initialize the variables
        init = tf.global_variables_initializer() #0.12
        sess.run(init)
            
        # Load pre-trained model weights
        saver = tf.train.Saver()
        if args.img_model != 'alexnet':
            checkpoint_file = '../models/checkpoints/'+args.img_model+'.ckpt'
        else:
            checkpoint_file = ''
        if checkpoint_file:
            saver.restore(sess, checkpoint_file)

        img_dir='../dataset/images/'
        # Start getting feature
        train_embedding = np.zeros((len(all_imgs), feat_dim), dtype=np.float32)
        bar = progressbar.ProgressBar(maxval=len(all_imgs), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i in range(len(all_imgs)):
            img = imread(os.path.join(img_dir, str(all_imgs[i]) + '.jpg'), mode='RGB')#.astype(np.float)
            img = imresize(img, (img_w, img_h))
            img = np.expand_dims(img, axis=0)
            train_dict = {my_net.img: img}
            img_feat = sess.run([my_net.img_feat], feed_dict=train_dict)
            id = all_imgs_dict.get(all_imgs[i],None)
            assert (id == i)
            train_embedding[id] = img_feat[0]
            bar.update(i)
        bar.finish()
        outFilePath = '../img_feat/%s' % (args.img_model)
        if not os.path.exists(outFilePath):
            os.makedirs(outFilePath)
        np.save(outFilePath+'/'+str(args.dataset)+'.npy', train_embedding)
        
            
if __name__ == '__main__':
    main()