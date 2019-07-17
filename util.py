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
import cv2
import math
import h5py
import pickle
import pandas as pd

def get_feat_from_image(image_path, save_flag, word_size, histogram_bin=64, image=None, save_path=None):
    # print(image)
    if image is None:
        image = cv2.imread(image_path)
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
        feat = calculate_feature(word, idx=i, save=save_flag, path=filename)
        hist = get_histogram(feat, nbins=histogram_bin)
        result[i,:] = hist
    if save_path is not None:
        pickle.dump(result, open(save_path, 'wb'))
    return result

def get_hist_from_image(image_path, kmeans, dict_size, word_size,
                        bag_size, overlap, save_flag, save_path):
    print(image_path)
    image = cv2.imread(image_path)
    image = np.array(image, dtype=int)
    bags = Bag(image, size=bag_size, overlap_pixel=overlap)
    result = np.zeros([bags.length, dict_size])
    for bag, i in bags:
        feat_words = get_feat_from_image(None, save_flag, word_size, image=bag)
        cluster_words = predict_kmeans(feat_words, kmeans)
        hist_bag = get_histogram(cluster_words, nbins=dict_size)
        result[i, :] = hist_bag
    pickle.dump(result, open(save_path, 'wb'))
    return result, bags



def load_mat(filename):
    f = h5py.File(filename, 'r')
    keys = list(f.keys())
    return np.array(f["I"]), np.array(f["M"])

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
        if (bag ==1).any():
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
    num_bag_w = int((w-overlap_pixel)/(size-overlap_pixel))
    box_h = int(math.floor(idx / num_bag_w) * (size-overlap_pixel))
    box_w = int(idx % (num_bag_w) * (size-overlap_pixel))

    return [box_h, box_h+size, box_w, box_w+size]

def calculate_label_from_bbox(dict_bbox, case_ID, w, length, factor, size=3600, overlap_pixel=2400):
    bboxes = dict_bbox[case_ID]
    result = np.zeros(length)
    for i in range(1, length):
        bb = bound_box(i, w, length, size, overlap_pixel)
        for bbox in bboxes:
            bbox = [int(x/4) for x in bbox]

            #if bb[0] >= bbox[0] and bb[1] <= bbox[1] and bb[2] >= bbox[2] and bb[3] <= bbox[3]:
            # if row overlaps
            if (bb[1] <= bbox[1] and bb[1] >= bbox[0]) or (bb[0] >= bbox[0] and bb[0] <= bbox[1]):
                # if col overlaps
                if (bb[3] <=bbox[3] and bb[3] >= bbox[2]) or (bb[2] >= bbox[2] and bb[2] <= bbox[3]):
                    result[i] = 1
                break
    return result

