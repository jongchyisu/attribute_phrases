import argparse

def train_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='', help='')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or eval')
    # Train-test set
    parser.add_argument('--train_set', type=str, default='train',
                        help='train or trainval')
    parser.add_argument('--test_set', type=str, default='val',
                        help='val or test')
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='32 for fine-tuning inception_v3 and vgg_16, use 128 otherwise')
    parser.add_argument('--learn_rate', type=float, default=0.001,
                       help='')
    parser.add_argument('--beta1', type=float, default=0.7,
                       help='beta1 for Adam optimizer')
    parser.add_argument('--max_steps', type=int, default=10000,
                       help='')
    parser.add_argument('--load_model_path', type=str, default='none',
                       help='model name (model-fixed-%steps) of the pre-trained whole model')
    # Loss
    parser.add_argument('--distance', type=str, default='dotProd',
                       help='dotProd, l2, l2sq, or cosine')
    parser.add_argument('--loss_fun', type=str, default='triplet',
                       help='triplet or contrastive')
    parser.add_argument('--loss_type', type=str, default='sen',
                       help='sen, img, or all (use sen for experiments)')
    parser.add_argument('--margin', type=float, default=5.0,
                       help='Margin for the loss function')
    parser.add_argument('--feat_dim', type=int, default=1024,
                       help='Feature dimension')
    # Image feature       
    parser.add_argument('--img_layer_depth', type=int, default=1,
                       help='only support 1 or 2')
    parser.add_argument('--img_model', type=str, default='vgg_16',
                       help='alexnet, inception_v3, or vgg_16')
    parser.add_argument('--train_img_model', type=int, default=0,
                       help='Fine-tune image model or not (default is False)')
    # Sentence feature
    parser.add_argument('--word_count_thresh', type=int, default=5,
                       help='Word count threshold for building vocabulary')
    parser.add_argument('--max_sent_length', type=int, default=9,
                       help='Max sentence length, smaller(9/17) for word, larger(65) for char')
    parser.add_argument('--word_embed_type', type=str, default='one-hot',
                       help='word2vec, one-hot, or char for character-based model')
    parser.add_argument('--bidirectional', type=int, default=0,
                       help='If use bidirectional, n_hidden will be halved')
    parser.add_argument('--state_or_output', type=str, default='state',
                       help='can use either state or output for rnn/lstm/gru cell')
    parser.add_argument('--word_layer_depth', type=int, default=1,
                       help='Number of cells in RNN')
    parser.add_argument('--sentence_model', type=str, default='lstm',
                       help='rnn, gru, or lstm, or cnn, or cnn-rnn')
    # Visualization
    parser.add_argument('--show_html', type=int, default=0,
                       help='Show wrong pairs and ranking result in html')
    # Others
    parser.add_argument('--ran_neg_sample', type=int, default=0,
                       help='Randomly sample negative example instead of pair data')
    parser.add_argument('--pairwise', type=int, default=0,
                       help='Use pairwise inputs: {I1,I2,S12,S21}')
    parser.add_argument('--dataset', type=str, default='train',
                       help='train, val or test')
    parser.add_argument('--input_path', type=str, default='',
                        help='Directory with model and config.json inside')
    parser.add_argument('--regularize', type=float, default=0.0001,
                        help='Regularization for fc layer')

    return parser