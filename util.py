#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 01:47:17 2019

@author: wuwenjun
"""

from bag import Bag
from word import Word
import numpy as np
import os
import cv2
import math
import h5py
import pandas as pd

def get_feat_from_image(image_path, save_flag, word_size, image=None, save_path=None):
    # print(image_path)
    if image is None:
        image = cv2.imread(image_path)
        image = np.array(image, dtype=int)
    
    words = Word(image, size=word_size)
    result = np.zeros([words.length, 320])

    for word, i in words:
        # get filename without extension
        if save_path is not None:
            dname = os.path.dirname(save_path)
            base = os.path.basename(im_p)
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
    return result        



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
    width = f['width']
    
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


def bound_box(self, idx):
    """
    Function that return the bounding box of a word given its index
    Args:
        ind: int, ind < number of words
    
    Returns:
        Bounding box(int[]): [h_low, h_high, w_low, w_high]
    """
    assert idx < self.length, "Index Out of Bound"
    num_bag_w = int((self.w-self.overlap_pixel)/(self.size-self.overlap_pixel))
    h = int(math.floor(idx / num_bag_w) * (self.size-self.overlap_pixel))
    w = int(idx % (num_bag_w) * (self.size-self.overlap_pixel))
    
    return [h, h+size, w, w+size]

def calculate_label(mask, size=3600, overlap_pixel=2400):
    bags = Bag(mask, size=size, overlap_pixel=overlap_pixel)
    label = np.zeros(bags.length)
    for bag, i in bags:
        if (bag ==1).any():
            label[i] = 1
        else:
            label[i] = 0
    return label


        
                


    
    