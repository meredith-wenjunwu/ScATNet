#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 01:05:00 2019

@author: wuwenjun
"""
from argparse import ArgumentParser
import cv2
import numpy as np
from feature import calculate_feature, get_histogram
from word import Word
from cluster import *
import os
import pickle


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--image', default='/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/test.jpg', help='Input image path')
    parser.add_argument('--save_intermediate', default=False, help='Whether or not to save the intermediate results')
    parser.add_argument('--dict_size', default= 40, help='Dictionary Size for KMeans')
    parser.add_argument('--histogram_bin', default=64, help='Bin size for histogram')
    parser.add_argument('--save_path', default='/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/output/test', help='save_path for results')
    
    args = parser.parse_args()
    
    image_path = args.image
    save_flag = args.save_intermediate
    dict_size = args.dict_size
    histogram_bin = args.histogram_bin
    save_path = args.save_path
    
    if image_path is not  None and save_path is not None:
        image = cv2.imread(image_path)
        image = np.array(image, dtype=int)
    
        words = Word(image)
        
        result = np.zeros([words.length, 320])
        
        for word, i in words:
            feat = calculate_feature(word, idx=i, save=save_flag, path=save_path)
            hist = get_histogram(feat, nbins=histogram_bin)
            result[i,:] = hist
        
    else:
        print('Error: Input image path is None or save path is None.')
        
    # Save features
    filename = save_path + '_feat.pkl'
    pickle.dump(result, open(filename, 'wb'))
    # K-Means Part
    
    filename = save_path + '_kmeans.pkl'
    kmeans = construct_kmeans(result)
    
    pickle.dump(kmeans, open(filename, 'wb'))
    
    
    