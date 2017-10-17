""" Used for generating user study on speaker output, 
    and for collecting user study result.    
"""


import json
import random
import time


def gather_userstudy_result():
    feedback_fname = 'user_study/user_study_1_res.txt'
    task_fnames = ['user_study/task_annotations_test_DS_1_100__2017-03-11-15.json',
                   'user_study/task_annotations_test_S_1_100__2017-03-11-15.json',
                   'user_study/task_annotations_val_DS_1_50__2017-03-11-15.json',
                   'user_study/task_annotations_val_S_1_50__2017-03-11-15.json']
    result_fnames = ['user_study/task_result_test_DS_1_100.json',
                     'user_study/task_result_test_S_1_100.json',
                     'user_study/task_result_val_DS_1_50.json',
                     'user_study/task_result_val_S_1_50.json']

    # Build result dict
    fb_dict = {}
    with open(feedback_fname) as result_f:
        for line in result_f:
            info = line.split(';')
            task_id = info[0]
            img1_name = info[1]
            img2_name = info[2]
            feed_back = int(info[3])
            task = fb_dict.get(task_id)
            if task is None:
                fb_dict[task_id] = {'img1_name': img1_name, 'img2_name': img2_name, 'feed_backs': [feed_back]}
            else:
                task['feed_backs'].append(feed_back)

    for tasks_i in range(len(task_fnames)):
        # update feed backs with ground truth
        task_fname = task_fnames[tasks_i]
        feedback_count = {}
        with open(task_fname) as task_f:
            task_annos = json.load(task_f)
        for task_id, task in task_annos['task_dict'].iteritems():
            assert fb_dict[task_id]['img1_name'] == task['img1_name']
            assert fb_dict[task_id]['img2_name'] == task['img2_name']
            raw_fb = fb_dict[task_id]['feed_backs']
            # previously it's 1 for img1, 0 for img2,
            # now change to 0 for img1, 1 for img2, to be same as user feedback
            ground_truth = 1 - task['ground_truth']
            correct_vote = 0
            incorrect_vote = 0
            feed_backs = [0, 0, 0]
            for i in range(3):
                if raw_fb[i] == 2:
                    feed_backs[i] = 0.1  # 0 for uncertain
                elif raw_fb[i] == ground_truth:
                    feed_backs[i] = 1  # 1 for correct answer
                    correct_vote += 1
                else:
                    feed_backs[i] = -1
                    incorrect_vote += 1
            if correct_vote >= 2:
                correctness = 1
            elif incorrect_vote >= 2:
                correctness = -1
            else:
                correctness = 0
            if feedback_count.get(str(sum(feed_backs))) is not None:
                feedback_count[str(sum(feed_backs))] += 1
            else:
                feedback_count[str(sum(feed_backs))] = 1

            fb_dict[task_id] = {'feed_backs': feed_backs, 'is_correct': correctness}

        print task_fname
        for score, count in feedback_count.iteritems():
            print score, count

        # add feed backs to annotations
        infer_anno_path = task_annos['infer_anno_path']
        with open(infer_anno_path) as infer_anno_f:
            infer_annos = json.load(infer_anno_f)
        sample_annos = []
        for task_id, task in task_annos['task_dict'].iteritems():
            case_id = task['case_id']
            infer_anno = infer_annos[case_id]
            assert infer_anno['id'] == case_id
            if task['ground_truth'] == 1:
                assert infer_anno['img1_id'] + '.jpg' == task['img1_name']
                assert infer_anno['img2_id'] + '.jpg' == task['img2_name']
            else:
                assert infer_anno['img1_id'] + '.jpg' == task['img2_name']
                assert infer_anno['img2_id'] + '.jpg' == task['img1_name']
            if infer_anno.get('feed_backs') is None:
                infer_anno['feed_backs'] = [None] * 10
                infer_anno['is_correct'] = [None] * 10
            text_i = task['text_id']
            infer_anno['feed_backs'][text_i] = fb_dict[task_id]['feed_backs']
            infer_anno['is_correct'][text_i] = fb_dict[task_id]['is_correct']
            if None not in infer_anno['is_correct']:
                sample_annos.append(infer_anno)
        result_fname = result_fnames[tasks_i]
        with open(result_fname, 'w') as result_f:
            json.dump(sample_annos, result_f)


def gen_task():
    for dataset_name in ['val', 'test']:
        if dataset_name == 'val':
            task_num = 50
        else:
            task_num = 100
        sample_ids = random.sample(range(2350), task_num)
        for speaker_mode in ['S', 'DS']:
            for text_mode in [0, 1]:
                if speaker_mode == 'S' and text_mode == 0:
                    continue

                infer_anno_path = 'output/%s_tuneCNN_model-40000__2017-03-07-19/' \
                                  'infer_annotations_%s_model-40000_case0_beam10_sent10.json' \
                                  % (speaker_mode, dataset_name)
                config_path = 'output/%s_tuneCNN_model-40000__2017-03-07-19/config.json' % speaker_mode
                task_fname = 'user_study/task_annotations_%s_%s_%d_%d__%s.json' \
                             % (dataset_name, speaker_mode, text_mode, task_num, time.strftime('%Y-%m-%d-%H'))
                txt_fname = 'user_study/task_input_%s_%s_%d_%d__%s.txt' \
                            % (dataset_name, speaker_mode, text_mode, task_num, time.strftime('%Y-%m-%d-%H'))
                task_dict = gen_task_set(infer_anno_path, config_path, dataset_name, speaker_mode, text_mode,
                                         task_fname, sample_ids)
                task_set_to_txt(task_dict, task_fname, txt_fname)


def gen_task_set(infer_anno_path, config_path, dataset_name, speaker_mode, text_mode, save_fname, sample_ids):
    """
    Args:
        infer_anno_path: inference annotation output file
        config_path: configuration file for getting the inference
        dataset_name: 'test' or 'val', which set the inference is on, should be consistent with infer_anno_path
        speaker_mode: 'S' or 'DS', should be consistent with infer_anno_path
        text_mode: 1: sent1; 2: sent2; 0: sent1 _VS sent2
        sample_ids: the ids of samples to use
    Returns:

    """
    with open(config_path) as config_file:
        config_dict = json.load(config_file)

    with open(infer_anno_path) as infer_anno_file:
        infer_annos = json.load(infer_anno_file)
    sample_annos = [infer_annos[i] for i in sample_ids]

    record_dict = {'config': config_dict,
                   'infer_anno_path': infer_anno_path,
                   'task_dict': dict()}
    for anno in sample_annos:
        case_id = anno['id']
        for text_id in range(len(anno['gen_texts'])):
            sent = get_sent_by_mode(anno['gen_texts'][text_id], text_mode)
            # if text_mode!=2, g_t before flip is 1 (img1), otherwise is 0 (img2)
            # if text_mode==2, g_t before flip is 0 (img2), otherwise is 1 (img1)
            ground_truth = int(text_mode != 2)
            flip = random.random() > 0.5
            if flip:
                ground_truth = 1 - ground_truth
                img1_name = str(anno['img2_id']) + '.jpg'
                img2_name = str(anno['img1_id']) + '.jpg'
            else:
                img1_name = str(anno['img1_id']) + '.jpg'
                img2_name = str(anno['img2_id']) + '.jpg'
            task = {'img1_name': img1_name,
                    'img2_name': img2_name,
                    'sentence': sent,
                    'ground_truth': ground_truth,
                    'case_id': case_id,
                    'text_id': text_id}
            task_id = '%s_%s_%d_%d_%d' % (speaker_mode, dataset_name, case_id, text_id, text_mode)
            record_dict['task_dict'][task_id] = task

    save_f = open(save_fname, 'w')
    json.dump(record_dict, save_f)
    save_f.close()
    return record_dict['task_dict']


def task_set_to_txt(task_dict, task_fname, out_fname):
    out_f = open(out_fname, 'w')
    out_f.write('** Generated from %s **\n' % task_fname)
    for task_id, task in task_dict.iteritems():
        msg = '%s;%s;%s;%s\n' % (task_id, task['img1_name'], task['img2_name'], task['sentence'])
        out_f.write(msg)
    out_f.close()


def get_sent_by_mode(text, text_mode):
    if text_mode == 0:
        return text
    sent_list = text.split('_VS')
    if text_mode == 1:
        return sent_list[0].strip()
    if text_mode == 2:
        sent2 = ''
        if len(sent_list) >= 2:
            sent2 = sent_list[1].strip()
        return sent2
    return None


if __name__ == "__main__":
    gather_userstudy_result()
