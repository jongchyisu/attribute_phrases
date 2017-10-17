# copied and modified from COCO, need to clean it!

# Interface for accessing the VisDiff dataset.

# The following API functions are defined:
#  VisDiff       - VisDiff api class that loads VisDiff annotation file and prepare data structures.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>annToMask, COCO>showAnns

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import time
import numpy as np
import copy
import itertools
import os
from collections import defaultdict
import sys
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve

class VisDiff:
    def __init__(self, annotation_file=None):
        """
        Constructor of VisDiff helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.imgs = dict(), dict(), dict()
        self.imgToAnns = defaultdict(list)
        if annotation_file is not None:
            #print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            #print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        #print 'creating index...'
        temp = self.dataset['annotations'][0]
        imgToAnns = {im: [] for im in self.dataset['img_all_id']}
        anns =      {ann['id']: [] for ann in self.dataset['annotations']}
        for ann in self.dataset['annotations']:
            imgToAnns[ann['img1_id']] += [ann['sentences1']]
            imgToAnns[ann['img2_id']] += [ann['sentences2']]
            anns[ann['id']] = ann

        imgs = {im: [] for im in self.dataset['img_all_id']}
        for img in self.dataset['img_all_id']:
            #imgs[img['id']] = img
            imgs[img] = img

        ''' not using category for now, see code in VisDiffDemo for this
        cats = []
        catToImgs = []
        if self.dataset['type'] == 'instances':
            cats = {cat['id']: [] for cat in self.dataset['categories']}
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat
            catToImgs = {cat['id']: [] for cat in self.dataset['categories']}
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']] += [ann['image_id']]
        '''
        #print 'index created!'

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        #self.catToImgs = catToImgs
        self.imgs = imgs
        #self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))

    def getAnnIds(self, imgIds=[], catIds=[], annIds=[]):
        """
        NOT USING catIds NOW!!
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]
        annIds = annIds if type(annIds) == list else [annIds]

        if len(imgIds) == len(catIds) == len(annIds) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                print imgIds
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
        ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        NOT USING THIS FUNCTION NOW!!
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if type(catNms) == list else [catNms]
        supNms = supNms if type(supNms) == list else [supNms]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        ids = ids if type(ids) == list else [ids]
        return [self.anns[id] for id in ids]
        #if type(ids) == list:
        #    return [self.anns[id] for id in ids]
        #elif type(ids) == int:
        #    return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if type(ids) == list:
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if type(ids) == list:
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def showAnns(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        for ann in anns:
            for i in range(len(ann['sentences1'])):
                print ann['sentences1'][i], '_vs_', ann['sentences2'][i]
            #print(ann['caption'])

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        #res = VisDiff()
        #res.dataset['annotations'] = [id for id in self.dataset['annotations']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str or type(resFile) == unicode:
            res = json.load(open(resFile))

        annsIds = [int(ann) for ann in res]
        assert set(annsIds) == (set(annsIds) & set(self.getAnnIds())), \
               'Results do not correspond to current coco set'
        #imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
        #res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
        #for id, ann in enumerate(anns):
        #    ann['id'] = id+1
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        #res.dataset['annotations'] = anns
        #res.createIndex()
        return res

