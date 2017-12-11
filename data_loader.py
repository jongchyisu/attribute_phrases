import numpy as np
import os, sys, random, re, string
from scipy.misc import imread, imresize
from collections import Counter
from pyVisDifftools.visdiff import VisDiff

# Special vocabulary symbols - we always put them at the start in the dictionary.
_PAD = b"_PAD"
_VS = b"_VS"
_EOS = b"_EOS"
_UNK = b"_UNK"
_STA = b"_STA"
_START_VOCAB = [_PAD, _VS, _EOS, _UNK, _STA]

PAD_ID = 0
VS_ID = 1
EOS_ID = 2
UNK_ID = 3
STA_ID = 4


def build_vocabulary(json_path='', word_count_thresh=0, word_embed_type='one-hot', save_path=None):
    """
    Build a vocabulary.
    Args:
        json_path: the dataset file used to build the vovabulary
        word_count_thresh: words which occur less than word_count_threshold times would be converted to UNK tokens
        save_path: if not None, save the vocabulary to npy file

    Returns:
        a vocabulary presented as a dictionary. Keys are word strings; values are word ids

    """
    if word_embed_type == 'char':
        annotations = VisDiff(json_path).dataset['annotations']
        all_text = ''
        for ann in annotations:
            for sent in ann['sentences1'] + ann['sentences2']:
                all_text += sent.lower()
        counter = Counter(all_text)
        for i,char in enumerate(_START_VOCAB):
            counter[char] = 100000000-i
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        chars, _ = zip(*count_pairs)
        vocab = dict(zip(chars, range(len(chars))))
            
    else:  # 'one-hot' encoding
        annotations = VisDiff(json_path).dataset['annotations']
        all_text = ''
        for ann in annotations:
            for sent in ann['sentences1'] + ann['sentences2']:
                all_text += ' ' + sent
        words = _tokenize_sentence(all_text, word_embed_type)

        counts = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1

        # vocabulary, keep words that occur often
        vocab_list = [w for w, n in counts.iteritems() if n > word_count_thresh]
        vocab_list = _START_VOCAB + vocab_list
        vocab = dict([(w, i) for (i, w) in enumerate(vocab_list)])

    if save_path is not None:
        np.save(save_path, vocab)
        print 'Vocabulary saved to: ' + save_path
    return vocab


def make_vocabularies_to_file(word_count_threshes, save_dir='vocabulary'):
    for t in word_count_threshes:
        s_path = os.path.join(save_dir, 'word_vocab_train_%d.npy' % t)
        build_vocabulary(word_count_thresh=t, save_path=s_path)


def _tokenize_sentence(sentence, word_embed_type='one-hot'):
    """
    Tokenize a sentence.

    Args:
    sentence: a sentence

    Returns:
    tokens: tokenized sentence, a list
    """
    if word_embed_type=='char':
        # Only do lower now, not sure about this...
        sentence_ = clean_str(sentence)
        tokens = [a for a in sentence_]
    else:
        sentence = re.sub("[^a-zA-Z0-9]", " ", sentence)
        tokens = str(sentence).lower().translate(None, string.punctuation).strip().split()
    # print tokens
    return tokens

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def split_sent_pair(text):
    sents = text.split('_VS')
    sent1 = sents[0].strip()
    sent2 = ''
    if len(sents) > 1:
        sent2 = sents[1].strip()
    return sent1, sent2


class DataFeeder:
    def __init__(self, annotations, vocabulary=None, train_img_model=False, img_dir='dataset/images/',
                 img_dict=None, feed_mode='DL', rand_neg=False, rand_flip=False, shuffle=True, trim_last_batch=False,
                 word_embed_type='one-hot', max_length=-1, img_w=224, img_h=224, img_c=3, init_epoch=0, augment=True):
        """
        Args:
            annotations: annotation dict of the dataset
            vocabulary: a dict to get word ids
            train_img_model: if True, feed img; if False, feed img id
            img_dir: dir to where the images are stored
            img_dict:
            feed_mode: 'S', 'DS'; 'SL', 'DL' (for 'Lr', use 'SL' with rand_neg=True)
            rand_neg: use random negative pair
            rand_flip: switch img1/2 and sent1/2 with p=50%
            shuffle: whether to shuffle the dataset before every epoch
            trim_last_batch: if False, use random samples to fill the last batch; if True, last batch is smaller
            word_embed_type: 'one-hot', 'word2vec', 'char'
            max_length: max length for text, including _STA, _VS, _EOF. Suggestion: 9 or 17
            img_w: img size
            img_h: img size
            img_c: img channel
            init_epoch: start number for counting epochs

        Returns:

        """

        self.vocabulary = vocabulary
        self.word_embed_type = word_embed_type
        self.feed_mode = feed_mode
        self.rand_neg = rand_neg
        self.rand_flip = rand_flip
        self.shuffle = shuffle
        self.trim_last_batch = trim_last_batch
        self.train_img_model = train_img_model
        self.img_dict = img_dict
        self.img_dir = img_dir

        self.max_length = max_length
        self.img_w = img_w
        self.img_h = img_h
        self.img_c = img_c
        self.epoch_count = init_epoch

        self.contain_probs = annotations[0].has_key('gen_probs')
        self.cur_idx = 0

        if self.contain_probs:
            self.annotations = self._separate_anno_rerank(annotations)
        elif self.feed_mode == 'S':
            self.annotations = self._separate_anno_single(annotations, augment)
        else:
            self.annotations = self._separate_anno(annotations)

        if shuffle:
            random.shuffle(self.annotations)
        
        if self.feed_mode == 'S':
            self.get_batch = self.get_batch_single
        else:
            self.get_batch = self.get_batch_pair
    
    def get_img(self, img_id, img_w, img_h):
        if self.train_img_model:
            # Note: no cropping and distortion for training for now
            img = imread(os.path.join(self.img_dir, str(img_id) + '.jpg'), mode='RGB')
            img = imresize(img, (img_w, img_h))
        else:  # lookup precomputed embedding
            img = self.img_dict.get(str(img_id), None)
        return img

    def get_batch_single(self, batch_size):
        assert self.feed_mode == 'S'
        assert not self.contain_probs
        # initialize
        if self.train_img_model:
            img1 = np.empty([batch_size, self.img_w, self.img_h, self.img_c])
        else:
            img1 = np.empty([batch_size], dtype=np.int32)
        sent1 = [[]] * batch_size
        # get images and sentences
        for i in range(0, batch_size):
            img1_id = self.annotations[self.cur_idx]['img_id']
            s1 = self.annotations[self.cur_idx]['sent']
            img1[i] = self.get_img(img1_id, img_w=self.img_w, img_h=self.img_h)
            sent1[i] = s1
            # update cur_idx, epoch_count
            self.cur_idx += 1
            if self.cur_idx >= len(self.annotations):
                self.cur_idx = 0
                self.epoch_count += 1
                if self.shuffle:
                    random.shuffle(self.annotations)
                if self.trim_last_batch:
                    img1 = img1[:i+1]
                    sent1 = sent1[:i+1]
                    break
        # padding and return
        encode_sent, sent_len = self._encode_pad_sentences(sent1, self.max_length)
        target_sent = self._input_to_target_text(encode_sent)
        sent_mask1 = self.len_to_mask(sent_len)
        batch_dict = {'img': img1,
                      'encode_sent': encode_sent,
                      'target_sent': target_sent,
                      'sent_mask': sent_mask1}
        return batch_dict

    def get_batch_pair(self, batch_size):
        # initialize
        if self.train_img_model:
            img1 = np.empty([batch_size, self.img_w, self.img_h, self.img_c])
            img2 = np.empty([batch_size, self.img_w, self.img_h, self.img_c])
        else:
            img1 = np.empty([batch_size], dtype=np.int32)
            img2 = np.empty([batch_size], dtype=np.int32)
        sent1 = [[]] * batch_size
        sent2 = [[]] * batch_size
        if self.contain_probs:
            gen_probs = np.empty(batch_size, dtype=np.float32)

        # get images and sentences
        for i in range(0, batch_size):
            img1_id = self.annotations[self.cur_idx]['img1_id']
            img2_id = self.annotations[self.cur_idx]['img2_id']
            s1 = self.annotations[self.cur_idx]['sent1']
            s2 = self.annotations[self.cur_idx]['sent2']
            if self.rand_flip and np.random.rand() > 0.5:
                t1 = img1_id
                img1_id = img2_id
                img2_id = t1
                t2 = s1
                s1 = s2
                s2 = t2
            if self.rand_neg:
                neg_sample = np.random.randint(0, len(self.annotations))
                if np.random.rand() > 0.5:
                    img1_id = self.annotations[neg_sample]['img1_id']
                    s1 = self.annotations[neg_sample]['sent1']
                else:
                    img2_id = self.annotations[neg_sample]['img2_id']
                    s2 = self.annotations[neg_sample]['sent2']
            img1[i] = self.get_img(img1_id, img_w=self.img_w, img_h=self.img_h)
            img2[i] = self.get_img(img2_id, img_w=self.img_w, img_h=self.img_h)
            sent1[i] = s1
            sent2[i] = s2
            if self.contain_probs:
                gen_probs[i] = self.annotations[self.cur_idx]['gen_probs']
            # update cur_idx, epoch_count
            self.cur_idx += 1
            if self.cur_idx >= len(self.annotations):
                self.cur_idx = 0
                self.epoch_count += 1
                if self.shuffle:
                    random.shuffle(self.annotations)
                if self.trim_last_batch:
                    img1 = img1[:i+1]
                    img2 = img2[:i+1]
                    sent1 = sent1[:i+1]
                    sent2 = sent2[:i+1]
                    if self.contain_probs:
                        gen_probs = gen_probs[:i+1]
                    break

        batch_dict = dict()
        if self.feed_mode == 'DS':
            encode_text, text_mask = self._concat_encode_pad_sentences(sent1, sent2, self.max_length)
            target_text = self._input_to_target_text(encode_text)
            batch_dict = {'img1': img1,
                          'img2': img2,
                          'encode_text': encode_text,
                          'target_text': target_text,
                          'text_mask': text_mask}
        elif self.feed_mode == 'SL':
            encode_sent1, sent_len1 = self._encode_pad_sentences(sent1, self.max_length)
            encode_sent2, sent_len2 = self._encode_pad_sentences(sent2, self.max_length)
            batch_dict = {'img1': img1,
                          'img2': img2,
                          'encode_sent1': encode_sent1,
                          'encode_sent2': encode_sent2,
                          'sent_len1': sent_len1,
                          'sent_len2': sent_len2}
        elif self.feed_mode == 'DL':
            encode_text1, _ = self._concat_encode_pad_sentences(sent1, sent2, self.max_length)
            encode_text2, text_mask = self._concat_encode_pad_sentences(sent2, sent1, self.max_length)
            batch_dict = {'img1': img1,
                            'img2': img2,
                            'encode_text1': encode_text1,
                            'encode_text2': encode_text2,
                            'text_len': np.sum(text_mask, 1)}
        if self.contain_probs:
            batch_dict['gen_probs'] = gen_probs

        return batch_dict

    @staticmethod
    def _separate_anno(anno):
        all_anno = []
        for i in xrange(len(anno)):
            for j in xrange(5):
                data = {'id': [],
                        'img1_id': [],
                        'img2_id': [],
                        'sent1': None,
                        'sent2': None}
                data['id'] = i*5+j
                data['img1_id'] = anno[i]['img1_id']
                data['img2_id'] = anno[i]['img2_id']
                data['sent1'] = anno[i]['sentences1'][j]
                data['sent2'] = anno[i]['sentences2'][j]
                all_anno.append(data)
        return all_anno

    @staticmethod
    def _separate_anno_rerank(anno):
        all_anno = []
        for i in xrange(len(anno)):
            for j in xrange(len(anno[i]['gen_sent1'])):
                data = {'img1_id': [],
                        'img2_id': [],
                        'sent1': None,
                        'sent2': None,
                        'gen_prob': 0}
                data['img1_id'] = anno[i]['img1_id']
                data['img2_id'] = anno[i]['img2_id']
                data['sent1'] = anno[i]['gen_sent1'][j]
                data['sent2'] = anno[i]['gen_sent2'][j]
                data['gen_probs'] = anno[i]['gen_probs'][j]
                all_anno.append(data)
        return all_anno

    @staticmethod
    def _separate_anno_single(anno, augment):
        all_anno = []
        for i in xrange(len(anno)):
            for j in xrange(5):
                data1 = {'img_id': [],
                         'sent': None}
                data1['img_id'] = anno[i]['img1_id']
                data1['sent'] = anno[i]['sentences1'][j]
                all_anno.append(data1)
                data2 = {'img_id': [],
                         'sent': None}
                data2['img_id'] = anno[i]['img2_id']
                data2['sent'] = anno[i]['sentences2'][j]
                if augment:
                    all_anno.append(data2)
        return all_anno

    def _encode_pad_sentences(self, sentences, sent_len_thresh):
        sent_len = np.empty(len(sentences), dtype=np.int32)
        encoded_sents = np.ones([len(sentences), sent_len_thresh], dtype=np.int32) * PAD_ID
        for i, sent in enumerate(sentences):
            words = [_STA] + _tokenize_sentence(str(sent), self.word_embed_type) + [_EOS]
            # cut to threshold length
            if len(words) > sent_len_thresh:
                words = words[0:sent_len_thresh]
            encoded_sents[i, 0: len(words)] = [self.vocabulary.get(w, UNK_ID) for w in words]
            sent_len[i] = len(words)
        return encoded_sents, sent_len

    def _concat_encode_pad_sentences(self, sent1, sent2, text_len_thresh):
        batch_size = len(sent1)
        sent_len_thresh = (text_len_thresh - 1) / 2
        concat_encode_text = np.ones((batch_size, text_len_thresh), dtype=np.int32) * PAD_ID
        text_mask = np.zeros((batch_size, text_len_thresh), dtype=np.int32)
        for i in range(batch_size):
            # tokenize
            s1 = [_STA] + _tokenize_sentence(sent1[i], self.word_embed_type)
            s2 = _tokenize_sentence(sent2[i], self.word_embed_type) + [_EOS]

            # cut to threshold length
            if len(s1) + len(s2) + 1 > text_len_thresh:
                if len(s1) <= sent_len_thresh:
                    s2 = s2[0: text_len_thresh - 1 - len(s1)]
                elif len(s2) <= sent_len_thresh:
                    s1 = s1[0: text_len_thresh - 1 - len(s2)]
                else:
                    s1 = s1[0: sent_len_thresh]
                    s2 = s2[0: sent_len_thresh]
            wds = s1 + [_VS] + s2

            # word to id
            concat_encode_text[i, 0:len(wds)] = [self.vocabulary.get(w, UNK_ID) for w in wds]
            text_mask[i, 0:len(wds)] = 1
        return concat_encode_text, text_mask

    @staticmethod
    def _input_to_target_text(input_text):
        target_text = np.empty(input_text.shape)
        target_text[:, 0: -1] = input_text[:, 1:]
        target_text[:, -1] = PAD_ID
        return target_text

    def len_to_mask(self, text_len):
        batch_size = np.shape(text_len)[0]
        text_mask = np.zeros((batch_size, self.max_length), dtype=np.int32)
        for i in range(batch_size):
            text_mask[i, 0:text_len[i]] = 1
        return text_mask


if __name__ == "__main__":
    make_vocabularies_to_file(range(6))
