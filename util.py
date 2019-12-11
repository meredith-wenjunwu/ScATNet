#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 01:47:17 2019

@author: wuwenjun
"""

from bag import Bag
from word import Word
from feature import calculate_feature, get_histogram, get_histogram_cluster
from cluster import predict_kmeans
import numpy as np
import os
import sys
import glob
import cv2
import math
import h5py
import pickle
import csv
import pandas as pd
import random
from re import search
from PIL import Image
try:
    import openslide
except (ImportError, OSError) :
    import warnings
    warnings.warn('Cannot use openslide function', ImportWarning)


def get_feat_from_image(image_path, save_flag, word_size,
                        histogram_bin=64, image=None, save_path=None):
    # print(image)
    if image is None:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert image is not None, "imread fail, check path"
        image = np.array(image, dtype=int)
    feat = calculate_feature(image)
    words = Word(feat, size=word_size)
    result = None

    for word, i in words:
        hist = get_histogram(word, nbins=histogram_bin)
        if result is None:
            result = np.zeros([words.length, len(hist)])
        result[i, :] = hist
    if save_path is not None:
        pickle.dump(result, open(save_path, 'wb'))
    return result


def check_empty(img, img_p=None):
    if img_p is not None:
        img = cv2.imread(img_p)
    img = np.array(img, dtype=np.uint8)
    im = Image.fromarray(img)
    colors = im.getcolors(im.size[0] * im.size[1])
    count = 0
    for c in colors:
        if all(p >= 220 for p in c[1]) and (max(c[1]) - min(c[1]) <= 10):
            count += c[0]
    return ((count / (im.size[0] * im.size[1]) >= 0.85) and
        (len(colors) / (im.size[0] * im.size[1]) < 0.001))

def get_hist_from_image(image_path, kmeans, hclusters, dict_size, word_size,
                        image=None):
    if image is None:
        feat_words = get_feat_from_image(image_path, False, word_size)
    else:
        feat_words = get_feat_from_image(None, False, word_size, image=image)
    cluster_words = predict_kmeans(feat_words, kmeans, h_cluster=hclusters)
    hist_bag = get_histogram_cluster(cluster_words, dict_size=dict_size)
    return hist_bag

def get_hist_from_large_image(image_path, kmeans, hclusters, bag_size,
                              dict_size, word_size, 
                              histogram_bin=64, 
                              overlap_pixel=0, image=None):
    if image is None:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert image is not None, "imread fail, check path"
        image = np.array(image, dtype=int)
    feat = calculate_feature(image)
    print('feature calculated...')
    print('size: {}'.format(feat.shape))
    bags = Bag(img=feat, size=bag_size,
               overlap_pixel=overlap_pixel, padded=True)
    result = np.zeros([len(bags), dict_size])
    for bag, i in bags:
        words = Word(bag, size=word_size)
        feat_words = None
        for word, j in words:
            if i == 0 and j == 0:
                print('feature word size: {}'.format(word.shape))
            hist = get_histogram(word, nbins=histogram_bin)
            if feat_words is None:
                feat_words = np.zeros([words.length, len(hist)])
            feat_words[j, :] = hist
        cluster_words = predict_kmeans(feat_words, kmeans,
                                       h_cluster=hclusters)
        hist_bag = get_histogram_cluster(cluster_words,
                                             dict_size=dict_size)
        print('hist of bag size: {}'.format(hist_bag))
        result[i,:] = hist_bag
    return result


def scale_result(result_pkl, factor, output_size, 
                           bag_size=3600, overlap=0):
    output = np.empty(output_size)
    label = pickle.load(open(result_pkl, 'rb'))
    bags = Bag(h=output_size[0], w=output_size[1],
               size=int(np.ceil(bag_size / factor)),
               overlap_pixel=int(np.ceil(overlap / factor)),
               padded=True)
    assert len(label) == len(bags), "size mismatch between output and input"

    for i in range(len(bags)):
        bbox = bags.bound_box(i)
        #Bounding box(int[]): [h_low, h_high, w_low, w_high]
        bbox[0] = max(0, min(bbox[0] - bags.top, output_size[0]))
        bbox[1] = max(0, min(bbox[1] - bags.top, output_size[0]))
        bbox[2] = max(0, min(bbox[2] - bags.left, output_size[1]))
        bbox[3] = max(0, min(bbox[3] - bags.left, output_size[1]))
        output[bbox[0]:bbox[1], bbox[2]:bbox[3]] = label[i]

    # print(np.unique(output))

    return output



def merge_result(path):
    """
    Function that merge cut result back
    """
    result_list = glob.glob(os.path.join(path, '*i*j*_result.pkl'))

    print(result_list)
    coord = []
    for x in result_list:
        basename = os.path.basename(x)
        no_extend = os.path.splitext(basename)[0]
        match = search('i\dj\d', no_extend)
        match = match.group(0)
        coord += [match]
    print(coord)

    coord_index = np.argsort(coord)
    last = result_list[coord_index[-1]]
    no_extend = os.path.splitext(last)[0]
    no_extend = no_extend.split('_result')[0]
    label_index_p = no_extend + '_label_index.pkl'
    label_index = pickle.load(open(label_index_p, 'rb'))

    out = np.ones(max(label_index) + 1)

    for x in coord_index:
        f = result_list[x]
        print(f)
        result = pickle.load(open(f, 'rb'))
        no_extend = os.path.splitext(f)[0]
        no_extend = no_extend.split('_result')[0]
        label_index_p = no_extend + '_label_index.pkl'
        label_index = pickle.load(open(label_index_p, 'rb'))
        out[label_index] = result

    outpath = os.path.join(path, os.path.basename(path) + '_result.pkl')
    pickle.dump(out, open(outpath, 'wb'))





def cut_large_image_and_label(image_path, image_label_path, image_size,
                              bag_size, overlap, output_path=None):
    """
    Function that cut large WSI and its corresponding label image
     into 4 pieces.
        Args:

        Returns:

    """
    assert os.path.exists(image_path)
    if output_path:
        filename, file_extension = os.path.splitext(image_path)
        basename = os.path.basename(filename)
        filename = os.path.join(output_path, basename)
    else:
        filename, file_extension = os.path.splitext(image_path)
        basename = os.path.basename(filename)
        if not os.path.exists(filename):
            os.mkdir(filename)
        filename = os.path.join(filename, basename)



    bags = Bag(h=image_size[0], w=image_size[1],
               size=bag_size, overlap_pixel=overlap,
               padded=True)
    label = pickle.load(open(image_label_path, 'rb'))
    h = bags.h
    w = bags.w
    assert len(bags) == len(label), "Length of bags and label don't match"
    num_bag_w = int(math.floor((w - overlap) / (bag_size -
                                                overlap)))
    #print(num_bag_w)
    num_bag_h = int(math.floor((h - overlap) / (bag_size -
                                                overlap)))
    #print(num_bag_h)

    # split the WSI to patches containing 10*10 bags
    # keep track of left upper corner
    h_i = 0
    w_i = 0

    # number of sub-patches
    num_h = int(math.ceil(num_bag_h / 5))
    num_w = int(math.ceil(num_bag_w / 5))

    for i in range(num_h):
        for j in range(num_w):
            left_top_corner_index = h_i * num_bag_w + w_i
            # subtract the padding
            left_top_corner_bbox = bags.bound_box(left_top_corner_index)
            left_top_corner_bbox[0] = max(0,
                                          left_top_corner_bbox[0] - bags.top)
            left_top_corner_bbox[1] = max(0,
                                          left_top_corner_bbox[1] - bags.top)
            left_top_corner_bbox[2] = max(0,
                                          left_top_corner_bbox[2] - bags.left)
            left_top_corner_bbox[3] = max(0,
                                          left_top_corner_bbox[3] - bags.left)

            #number of rows
            right_row = min(num_bag_h - 1, h_i + 4)
            right_col = min(num_bag_w - 1, w_i + 4)
            right_bot_corner_index = int(min(len(bags),
                                             (right_row) * num_bag_w +
                                             right_col))
            right_bot_corner_bbox = bags.bound_box(right_bot_corner_index)
            # subtract the padding
            right_bot_corner_bbox = bags.bound_box(right_bot_corner_index)
            right_bot_corner_bbox[0] = max(0,
                                          right_bot_corner_bbox[0] - bags.top)
            right_bot_corner_bbox[1] = max(0,
                                           right_bot_corner_bbox[1] - bags.top)
            right_bot_corner_bbox[2] = max(0,
                                        right_bot_corner_bbox[2] - bags.left)
            right_bot_corner_bbox[3] = max(0,
                                          right_bot_corner_bbox[3] - bags.left)

            bbox = [left_top_corner_bbox[0], right_bot_corner_bbox[1],
                    left_top_corner_bbox[2], right_bot_corner_bbox[3]]

            label_patch = np.zeros([right_row - h_i + 1,
                                    right_col - w_i + 1])
            label_ind = [x * num_bag_w + y
                         for x in range(h_i, right_row + 1)
                         for y in range(w_i, right_col + 1)]
            label_patch = label[label_ind]
            im_outname = filename + '_i{}j{}'.format(i, j) + file_extension
            label_outname = filename + '_i{}j{}_label'.format(i, j) + '.pkl'
            label_index_outname = filename + '_i{}j{}_label_index'.format(i, j)+ '.pkl'
            pickle.dump(label_patch, open(label_outname, 'wb'))
            pickle.dump(label_ind, open(label_index_outname, 'wb'))
            if 'openslide' in sys.modules:
                crop_bbox_single_openslide(image_path, bbox, im_outname)
            else:
                crop_bbox_single(image_path, bbox, im_outname)
            w_i = min(num_bag_w - 1, w_i + 5)
        w_i = 0
        h_i = min(num_bag_h - 1, h_i + 5)



def load_mat(filename):
    f = h5py.File(filename, 'r')
    keys = list(f.keys())
    im = np.array(f["I"])
    im = np.transpose(im, [2, 1, 0])
    # im = np.flip(im, 2)
    return im, np.array(f["M"]).T


def preprocess_roi_csv(csv_file):
    header = None
    case=y=x=width=heigh=0

    result = {}
    with open(csv_file, newline='') as f:
        f.seek(0)
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if i == 0:
                header = row
                assert 'Case ID' in header, "No matching column with name: Case ID"
                assert 'Y' in header, "No matching column with name: Y"
                assert 'X' in header, "No matching column with name: X"
                assert 'Width' in header, "No matching column with name: Width"
                assert 'Height' in header, "No matching column with name: Height"
                case = header.index('Case ID')
                y = header.index('Y')
                x = header.index('X')
                width = header.index('Width')
                height = header.index('Height')
            if i > 0:
                try:
                    Y = int(row[y])
                except ValueError:
                    Y = int(float(row[y]))
                try:
                    X = int(row[x])
                except ValueError:
                   X = int(float(row[x]))
                c = int(row[case])
                try:
                    w = int(row[width])
                except ValueError:
                    w = int(float(row[width]))
                try:
                    h = int(row[height])
                except ValueError:
                    h = int(float(row[height]))
                bbox = [Y, Y+h, X, X+w]
                bb_l = result.get(c)
                if bb_l is None:
                    result[c] = [bbox]
                else:
                    result[c].append(bbox)
            i += 1
    return result


def get_immediate_subdirectories(a_dir):
    out = [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
    out.sort()
    return out

def preprocess_wsi_size_csv(csv_file):
    f = pd.read_csv(csv_file)
    caseID = f['Case ID']
    H = f['H']
    W = f['W']

    result = {}
    i = 0
    while i < len(caseID):
        # row_up, row_down, col_left, col_right

        height = H[i]
        width = W[i]
        bb_l = result.get(caseID[i])
        if bb_l is None:
            result[int(caseID[i])] = [height, width]
        i += 1
    return result


def bound_box(idx, w, length, size, overlap_pixel):
    """
    Function that return the bounding box of a word given its index
    Args:
        ind: int, ind < number of words

    Returns:
        Bounding box(int[]): [h_low, h_high, w_low, w_high]
    """
    assert idx < length, "Index Out of Bound"
    num_bag_w = int((w - overlap_pixel) / (size - overlap_pixel))
    box_h = int(math.floor(idx / num_bag_w) * (size - overlap_pixel))
    box_w = int(idx % (num_bag_w) * (size - overlap_pixel))

    return [box_h, box_h + size, box_w, box_w + size]


def calculate_label_from_roi_bbox(roi_bbox, wsi_size, factor=1,
                                  size=3600, overlap_pixel=2400):

    bags = Bag(h=wsi_size[0], w=wsi_size[1],
               size=size, overlap_pixel=overlap_pixel,
               padded=True)

    h, w = bags.h, bags.w
    pos_ind = set()
    num_bag_w = int(math.floor((w - overlap_pixel) / (size - overlap_pixel)))
    num_bag_h = int(math.floor((h - overlap_pixel) / (size - overlap_pixel)))
    # Bounding box(int[]): [h_low, h_high, w_low, w_high]
    for h_low, h_high, w_low, w_high in roi_bbox:
        h_low += bags.top
        h_high += bags.top
        w_low += bags.left
        w_high += bags.left
        #if h_high >= h and w_high >= w:
        if h_high > h or w_high > w:
            print("Size incompatible for case: {}".format(self.caseID))
            print("Bounding box: {}, {}, {}, {}".format(h_low, h_high, w_low, w_high))
            print("WSI size: {}, {}". format(h, w))
            h_high = min(h, h_high)
            w_high = min(w, w_high)
        ind_w_low = int(max(math.floor((w_low - size) / (size - overlap_pixel) +
           1), 0))
        ind_w_high = int(min(max(math.floor(w_high / (size - overlap_pixel)),
           0), num_bag_w - 1))
        ind_h_low = int(max(math.floor((h_low - size) / (size - overlap_pixel) +
           1), 0))
        ind_h_high = int(min(max(math.floor(h_high / (size - overlap_pixel)),
            0), num_bag_h - 1))
        for i in range(ind_h_low, ind_h_high + 1):
            pos_ind.update(range(i * num_bag_w + ind_w_low,
               i * num_bag_w + ind_w_high + 1))
    pos_ind_copy = list(pos_ind)

    for ind in pos_ind:
        h_low, h_high, w_low, w_high = bags.bound_box(ind)
        h_low -= bags.top
        h_high -= bags.top
        w_low -= bags.left
        w_high -= bags.left
        if not checkROI([h_low, h_high, w_low, w_high], roi_bbox):
            pos_ind_copy.remove(ind)
    pos_ind = np.sort(list(pos_ind_copy))
    result = np.zeros(len(bags))
    result[pos_ind] = 1
    return result



def biggest_bbox(bbox_list):
    row_low = 1000000000000
    row_high = -1
    col_low = 1000000000000
    col_high = -1
    for bbox in bbox_list:
        row_low = (bbox[0] if bbox[0] < row_low else row_low)
        row_high = (bbox[1] if bbox[1] > row_high else row_high)
        col_low = (bbox[2] if bbox[2] < col_low else col_low)
        col_high = (bbox[3] if bbox[3] > col_high else col_high)
    return [row_low, row_high, col_low, col_high]


def crop_saveroi_batch(image_folder, dict_bbox, appendix='.jpg'):
    f_ls = glob.glob(os.path.join(image_folder, '*.tif'))
    for f in f_ls:
        base = os.path.basename(f)
        name_noextend = os.path.splitext(base)[0]
        outname = os.path.join(image_folder, 'roi', name_noextend + appendix)
        if not os.path.exists(outname):
            caseID = int(name_noextend.split('_')[0][1:])
            bboxes = dict_bbox[caseID]
            bbox_final = biggest_bbox(bboxes)
            size_r = bbox_final[1] - bbox_final[0]
            size_c = bbox_final[3] - bbox_final[2]
            args = 'convert ' + f + ' -crop ' + str(size_c) + 'x' + str(
                size_r) + '+' + str(bbox_final[2]) + '+' + str(bbox_final[0]) + ' ' + outname
            #print(args)
            os.system(args)

def crop_bbox_single(image, bbox, outname):
    size_r = bbox[1] - bbox[0]
    size_c = bbox[3] - bbox[2]
    args = 'convert ' + image + ' -crop ' + str(size_c) + 'x' + str(size_r) + '+' + str(bbox[2]) + '+' + str(bbox[0]) + ' ' + outname
    print(args)
    os.system(args)


def crop_bbox_single_openslide(image, bbox, outname=None, level=0):
    assert 'openslide' in sys.modules, "Check Openslide installation"
    size_r = bbox[1] - bbox[0]
    size_c = bbox[3] - bbox[2]
    top_left = (bbox[2], bbox[0])
    slide = openslide.OpenSlide(image)
    crop = slide.read_region(top_left, level, (size_c, size_r)).convert('RGB')
    if outname is None:
        return crop
    else:
        crop.save(outname)



def cross_valid_index(positives, negatives, n_splits=10):
    kf = KFold(n_splits=ns)
    train_pos, test_pos = kf.split(positives)
    train_neg, test_neg = kf.split(negatives)

    train_pos_set = [positives[x] for x in train_pos]
    test_pos_set = [positives[x] for x in test_pos]
    train_neg_set = [negatives[x] for x in train_neg]
    test_neg_set = [negatives[x] for x in test_neg]

    train = train_pos_set + train_neg_set
    train_L = [1] * len(train_pos_set) + [0] * (train_neg_set)

    test = test_pos_set + test_neg_set
    test_L = [1] * len(test_pos_set) + [0] * len(test_neg_set)
    return train, train_L, test, test_L


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0, 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return intersection_area, iou


def checkROI(idx_bbox, bboxes, window_size=3600):
    [row_up, row_down, col_left, col_right] = idx_bbox
    check_bbox = {'x1': col_left, 'y1': row_up,
                  'x2': col_right, 'y2': row_down}
    check_area = 0
    for bb in bboxes:
        [row_up, row_down, col_left, col_right] = bb

        roi_bbox = {'x1': col_left, 'y1': row_up,
                  'x2': col_right, 'y2': row_down}
        intersection_area, perc = get_iou(roi_bbox, check_bbox)
        check_area += intersection_area
    if check_area / (window_size ** 2) < 0.6:
        return False
    return True


class ROI_Sampler:
    def __init__(self, caseID, window_size,
                 overlap, outdir, wsi_path=None,
                 roi_csv=None, wsi_size_csv=None,
                 dict_bbox=None, dict_wsi_size=None):
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        if dict_bbox is None:
            assert os.path.exists(roi_csv), "ROI csv file do not exist"
            self.dict_bbox = preprocess_roi_csv(roi_csv)
        if dict_wsi_size is None:
            assert os.path.exists(wsi_size_csv), "ROI size csv file do not exist"
            self.dict_wsi_size = preprocess_wsi_size_csv(wsi_size_csv)
        self.bboxes = self.dict_bbox[caseID]
        self.caseID = caseID
        self.wsi_size = dict_wsi_size[caseID]
        self.wsi_path = wsi_path
        self.window_size = window_size
        self.overlap = overlap
        self.bags = None
        self.pos_bags = None
        self.count = None
        self.pos_count = None
        self.neg_count = None
        self.outdir = outdir
        #self.negdir = os.path.join(self.outdir, 'neg')

    def sample_pos(self):
        # assert os.path.exists(self.roi_mat), "ROI mat file do not exist"
        self.negdir = os.path.join(self.outdir, 'neg')
        if not os.path.exists(self.negdir): os.mkdir(self.negdir)
        self.posdir = os.path.join(self.outdir, 'pos')
        if not os.path.exists(self.posdir): os.mkdir(self.posdir)
        # self.bags, self.pos_bags, self.count = self._sample_from_ROI_mat(self.roi_mat)

        bags = Bag(h=self.wsi_size[0], w=self.wsi_size[1],
                   size=self.window_size, overlap_pixel=2400,
                   padded=False)
        pos_ind = self._bbox_to_bags_ind_in_wsi(self.bboxes, self.wsi_size,
                                                self.window_size,
                                                2400)
        result = list(pos_ind)
        for ind in pos_ind:
            if not checkROI(bags.bound_box(ind), self.bboxes):
                result.remove(ind)
        pos_ind = np.sort(result)
        print("positive samples from {}: {}".format(self.caseID,
                                                    len(pos_ind)))

        print('roi bags index: {}'.format(pos_ind))
        for i in pos_ind:
            outname = os.path.join(self.posdir, str(self.caseID) +
                        '_' + str(i) + '.jpg')
            if not os.path.exists(outname):
                if 'openslide' in sys.modules:
                    crop_bbox_single_openslide(self.wsi_path,
                                               bags.bound_box(i), outname)
                else:
                    crop_bbox_single(self.wsi_path,
                                     bags.bound_box(i), outname)

        self.pos_count = len(pos_ind)
        self.neg_count = max(self.pos_count, 5)
        # print("nagative samples from " + str(self.caseID) + ' :' + str
        #     (self.neg_count))
        # self.neg_count = self.pos_count - self.neg_count

    def sample_neg(self, neg_count=None, mode=None):
        mode_type = ['rand', 'relevant']
        assert mode in mode_type, "Enter valid mode type"
        bags = Bag(h=self.wsi_size[0], w=self.wsi_size[1],
                   size=self.window_size, overlap_pixel=0,
                   padded=False)

        if not neg_count:
            neg_count = self.neg_count
            assert self.neg_count is not None, "Need to run sample_pos first"
        if mode == 'rand':
            self.neg_bags = self._sample_negative_samples_rand(neg_count,
                                                                self.bboxes,
                                                                bags)
        else:
            pos_ind = self._bbox_to_bags_ind_in_wsi(self.bboxes, self.wsi_size,
                                                     self.window_size,
                                                     0)

            self.neg_bags = self._sample_negative_samples_relevant(neg_count,
                self.wsi_size, pos_ind, self.window_size, bags)

            print(self.neg_bags)

            if self.wsi_path is not None:
                # need to crop roi and save in negdir
                for ind in self.neg_bags:
                    bbox = bags.bound_box(ind)
                    outname = str(self.caseID) + '_' + str(ind) + '.jpg'
                    outname = os.path.join(self.negdir, outname)
                    if not os.path.exists(outname):
                        if 'openslide' in sys.modules:
                            crop_bbox_single_openslide(self.wsi_path, bbox,
                                                       outname)
                        else:
                            crop_bbox_single(self.wsi_path, bbox, outname)

    def _bbox_to_bags_ind_in_wsi(self, bboxes, WSI_size, window_size, overlap):
        """
            This function calculates the ROI index in terms of window(bag/word)
            sizes (i.e. given the size of WSI, give out a list with the index of
            bags that are contained in the ROI)

            Args:
                bboxes (List(n)): list of bounding boxes of a given image
                WSI_size [h (int), w (int)]: size of the WSI (height, width)
                window size (int): size of word/bags or any window of interest
                                  (usually
                                  3600 for bags and 120 for words)
                overlap (int): overlapping pixel in window

            Returns:
                result (List(m)): list with the index of bags that are contained in
                the ROI
        """
        # assumption is that we won't receive anything on the border where bags
        # can't fit
        h, w = WSI_size
        result = set()
        num_bag_w = int(math.floor((w - overlap) / (window_size - overlap)))
        num_bag_h = int(math.floor((h - overlap) / (window_size - overlap)))
        # Bounding box(int[]): [h_low, h_high, w_low, w_high]
        for h_low, h_high, w_low, w_high in bboxes:
            #if h_high >= h and w_high >= w:
            if h_high > h or w_high > w:
                print("Size incompatible for case: {}".format(self.caseID))
                print("Bounding box: {}, {}, {}, {}".format(h_low, h_high, w_low, w_high))
                print("WSI size: {}, {}". format(h, w))
                h_high = min(h, h_high)
                w_high = min(w, w_high)
            ind_w_low = int(max(math.floor((w_low - window_size) / (window_size
               - overlap) + 1), 0))
            ind_w_high = int(min(max(math.floor(w_high / (window_size -
                overlap)), 0), num_bag_w - 1))
            ind_h_low = int(max(math.floor((h_low - window_size) / (window_size
               - overlap) + 1), 0))
            ind_h_high = int(min(max(math.floor(h_high / 
                (window_size - overlap)), 0), num_bag_h - 1))
            for i in range(ind_h_low, ind_h_high + 1):
                result.update(range(i * num_bag_w + ind_w_low,
                    i * num_bag_w + ind_w_high + 1))
        return np.sort(list(result))


    def _sample_negative_samples_relevant(self, num_of_neg_samples,
                                           WSI_size, roi_bags,
                                           window_size, bags):
        """
            This function calculates sample negative samples next to the ROI

            Args:
                bboxes (List(n)): list of bounding boxes of a given image
                WSI_size [h (int), w (int)]: size of the WSI (height, width)
                roi_bags (List(m)): list with the index of roi bags that are
                contained
                                    in the ROI (Result from
                                    bbox_to_bags_ind_in_wsi)
                window size (int): size of word/bags or any window of interest
                                  (usually
                                  3600 for bags and 120 for words)
                overlap (int): overlapping pixel in window

            Returns:
                result (List(m)): List of index of sampled negative bags
        """
        assert roi_bags is not None and len(roi_bags) > 0, "invalid roi bags"
        h, w = WSI_size
        overlap = 0
        num_bag_w = int(math.floor((w - overlap) / (window_size - overlap)))
        length = math.floor(math.floor((h - overlap) / (window_size - overlap)) *
           math.floor((w - overlap) / (window_size - overlap)))
        count_left = num_of_neg_samples
        result = set()
        ind_list = list(range(length))

        i = 0
        roi_bags = set(roi_bags)
        checked = set()

        while count_left > 0:
            # goes clockwise to sample bags
            if len(roi_bags) <= 0: 
                print("ROI too big, not enough negative sample")
                print("Number of negatives sampled/Needed: {}/{}".format(len(result), num_of_neg_samples))
                break
            ind = roi_bags.pop()
            # print('checking: {}'.format(ind))
            if ind not in checked:
                checked.add(ind)
                neigh = self._ROI_neighbor_not_roi(ind,
                                                   num_bag_w, length, bags)
                if count_left >= len(neigh):
                    result.update(neigh)
                    
                else:
                    neigh = random.sample(neigh, num_of_neg_samples - len(result))
                    result.update(neigh)
                count_left = num_of_neg_samples - len(result)
                for n in neigh:
                    if n not in checked:
                        roi_bags.add(n)
            #roi_bags.extend(neigh)
        print('---------------------negative bags index------------------')
        return list(result)



    def _checkROI(self, idx, idx_bbox, length):

        if idx < 0 or idx >= length:
            return False
        [row_up, row_down, col_left, col_right] = idx_bbox
        check_bbox = {'x1': row_up, 'y1': col_left,
                    'x2': row_down, 'y2': col_right}
        check_area = 0
        for bb in self.bboxes:
            [row_up, row_down, col_left, col_right] = bb

            roi_bbox = {'x1': row_up, 'y1': col_left,
                        'x2': row_down, 'y2': col_right}
            intersection_area, perc = get_iou(roi_bbox, check_bbox)
            check_area += intersection_area
        if check_area / (self.window_size ** 2) > 0.2:
            return False
        return True







    def _ROI_neighbor_not_roi(self, idx, num_bag_w, length, bags):
        result = []
        if idx < length and self._checkROI(idx, bags.bound_box(idx),
                                           length):
            result += [int(idx)]
        if idx + 1 < length and self._checkROI(idx + 1,
                                               bags.bound_box(idx + 1),
                                               length):
            result += [int(idx + 1)]
        if idx - 1 > 0 and self._checkROI(idx - 1,
                                          bags.bound_box(idx-1),
                                          length):
            result += [int(idx - 1)]
        if idx - num_bag_w > 0 and self._checkROI(idx - num_bag_w,
                                                  bags.bound_box(idx -
                                                   num_bag_w),
                                                  length):
            result += [int(idx - num_bag_w)]
        if idx + num_bag_w < length and self._checkROI(idx + num_bag_w,
                          bags.bound_box(idx + num_bag_w), length):
            result += [int(idx + num_bag_w)]
        return result



    def _sample_negative_samples_rand(self, num_of_neg_samples, bboxes, bags):
        count_left = num_of_neg_samples
        ind_list = list(range(len(bags)))
        bbox = biggest_bbox(bboxes)
        result = np.zeros(num_of_neg_samples)
        while count_left > 0:
            i = np.random.choice(ind_list)
            ind_list.remove(i)
            num_intersected_pixel = 0

            # if bb[0] >= bbox[0] and bb[1] <= bbox[1] and bb[2] >= bbox[2] and bb[3] <= bbox[3]:
            # if row overlaps
            bb = bags.bound_box(i)
            roi_row = range(bbox[0], bbox[1])
            sample_row = set(range(bb[0], bb[1]))
            intersect_row = sample_row.intersection(roi_row)

            if len(intersect_row) > 0:
                # if col overlaps
                roi_col = range(bbox[2], bbox[3])
                sample_col = set(range(bb[2], bb[3]))
                intersect_col = sample_col.intersection(roi_col)

                num_intersected_pixel += len(intersect_col) * len(intersect_row)
                if num_intersected_pixel <= 0.2 * size:
                    result[num_of_neg_samples - count_left] = i
                    count_left -= 1
        return bags[result]
