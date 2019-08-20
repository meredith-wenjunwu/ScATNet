#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 01:47:17 2019

@author: wuwenjun
"""

from bag import Bag
from word import Word
from feature import calculate_feature, get_histogram
import numpy as np
import os
import glob
import cv2
import math
import h5py
import pickle
import pandas as pd
from sklearn.model_selection import KFold


def get_feat_from_image(image_path, save_flag, word_size,
                        histogram_bin=64, image=None, save_path=None):
    # print(image)
    if image is None:
        image = cv2.imread(image_path)
        assert image is not None, "imread fail, check path"
        image = np.array(image, dtype=int)
    words = Word(image, size=word_size)
    result = np.zeros([words.length, 320])

    for word, i in words:
        # get filename without extension
        if save_path is not None:
            dname = os.path.dirname(save_path)
            base = os.path.basename(image_path)
            path_noextend = os.path.splitext(base)[0]
            filename = os.path.join(dname, path_noextend)
        else:
            filename = None
        feat = calculate_feature(word, idx=i, save=save_flag, path=filename)
        hist = get_histogram(feat, nbins=histogram_bin)
        result[i, :] = hist
    if save_path is not None:
        pickle.dump(result, open(save_path, 'wb'))
    return result


def get_hist_from_image(bags, kmeans, hclusters, dict_size, word_size,
                        save_path):
    result = np.zeros([bags.length, dict_size])
    for bag, i in bags:
        feat_words = get_feat_from_image(None, False, word_size, image=bag)
        cluster_words = predict_kmeans(feat_words, kmeans, h_cluster=hclusters)
        hist_bag = get_histogram(cluster_words, nbins=dict_size)
        result[i, :] = hist_bag
    pickle.dump(result, open(save_path, 'wb'))
    return result, bags


def load_mat(filename):
    f = h5py.File(filename, 'r')
    keys = list(f.keys())
    im = np.array(f["I"])
    im = np.transpose(im, [2, 1, 0])
    im = np.flip(im, 2)
    return im, np.array(f["M"])


def preprocess_roi_csv(csv_file):
    f = pd.read_csv(csv_file)
    caseID = f['Case ID']
    Y = f['Y']
    X = f['X']

    height = f['Height']
    width = f['Width']

    result = {}

    i = 0
    while i < len(caseID):
        # row_up, row_down, col_left, col_right

        bbox = [Y[i], Y[i] + height[i], X[i], X[i] + width[i]]
        bb_l = result.get(caseID[i])
        if bb_l is None:
            result[caseID[i]] = [bbox]
        else:
            result[caseID[i]].append(bbox)
        i += 1
    return result


def calculate_label_from_mask(mask, size=3600, overlap_pixel=2400):
    bags = Bag(mask, size=size, overlap_pixel=overlap_pixel)
    label = np.zeros(bags.length)
    for bag, i in bags:
        if (bag == 1).any():
            label[i] = 1
        else:
            label[i] = 0
    return label


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


def calculate_label_from_bbox(dict_bbox, case_ID, w, length, factor, size=3600, overlap_pixel=2400):
    bboxes = dict_bbox[case_ID]
    result = np.zeros(length)
    for i in range(1, length):
        bb = bound_box(i, w, length, size, overlap_pixel)
        for bbox in bboxes:
            bbox = [int(x / 4) for x in bbox]

            # if bb[0] >= bbox[0] and bb[1] <= bbox[1] and bb[2] >= bbox[2] and bb[3] <= bbox[3]:
            # if row overlaps
            if (bb[1] <= bbox[1] and bb[1] >= bbox[0]) or (bb[0] >= bbox[0] and bb[0] <= bbox[1]):
                # if col overlaps
                if (bb[3] <= bbox[3] and bb[3] >= bbox[2]) or (bb[2] >= bbox[2] and bb[2] <= bbox[3]):
                    result[i] = 1
                break
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


def crop_saveroi(image_folder, dict_bbox, appendix='.jpg'):
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
            print(args)
            os.system(args)


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


def sample_from_roi_mat(mat_filename):
    im, M = load_mat(mat_filename)
    words = Word(im)
    bags = Bag(words.img, padded=False)
    result = np.zeros(len(bags))
    pos_count = 0
    neg_count = 0
    for bag, i in bags:
        bbox = bags.bound_box(i)
        r, c = bag.shape
        size = r * c
        if np.sum(M[bbox[0]:bbox[1], bbox[2]:bbox[3], :]) / size >= 0.7:
            result[i] = 1
            pos_count += 1
        else:
            neg_count += 1
    return bags, result, [pos_count, neg_count]


def bbox_to_bags_ind_in_wsi(bboxes, [h, w], window_size, overlap):
    """
        This function calculates the ROI index in terms of window(bag/word)
        sizes (i.e. given the size of WSI, give out a list with the index of
        bags that are contained in the ROI)

        Args:
            bboxes (List(n)): list of bounding boxes of a given image
            [h (int), w (int)]: size of the WSI (height, width)
            window size (int): size of word/bags or any window of interest 
            (usually
            3600 for bags and 120 for words)
            overlap (int): overlapping pixel in window
    """
    # assumption is that we won't receive anything on the border where bags
    # can't fit
    result = set()
    num_bag_w = math.floor((w - overlap) / (window_size - overlap))
    # Bounding box(int[]): [h_low, h_high, w_low, w_high]
    for h_low, h_high, w_low, w_high in bboxes:
        assert h_high <= h and w_high <= w, "Size incompatible"
        ind_w_low = max(math.ceil((w_low - window_size) / (window_size -
           overlap)), 0)
        ind_w_high = max(math.ceil((w_high - window_size) / (window_size -
           overlap)), 0)
        ind h_low = max(math.ceil(h_low - window_size) / (window_size -
           overlap), 0)
        ind_h_high = max(math.ceil(h_high - window_size) / (window_size -
            overlap), 0)
        for i in range(h_low, h_high+1):
            result.update(range(h_low * num_bag_w + w_low, h_low * num_bag_w
                + w_high + 1))
    return list(result)




def sample_negative_samples(num_of_neg_samples, bboxes, bags):
    # need to sample negative samples next to the ROI
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

        if len(intersect_list) > 0:
            # if col overlaps
            roi_col = range(bbox[2], bbox[3])
            sample_col = set(range(bb[2], bb[3]))
            intersect_col = sample_col.intersection(roi_col)

            num_intersected_pixel += len(intersect_col) * len(intersect_row)
            if num_intersected_pixel <= 0.2 * size
                result[num_of_neg_samples - count_left] = i
                count_left -= 1
    return bags, result
