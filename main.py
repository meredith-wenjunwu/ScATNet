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
import glob
from bag import Bag

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--mode',
                        default='kmeans',
                        const='kmeans',
                        nargs='?',
                        choices=['kmeans', 'kmeans_visual', 'classifier_train', 'classifier_test'],
                        help='Choose mode from k-means clustering, visualization and classification_training, classification_testing')
    parser.add_argument('--single_image', default='/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/test.jpg', help='Input image path')
    parser.add_argument('--image_folder', default=None, help='Input image batch folder' )
    parser.add_argument('--save_intermediate', default=False, help='Whether or not to save the intermediate results')
    parser.add_argument('--dict_size', default= 40, help='Dictionary Size for KMeans')
    parser.add_argument('--histogram_bin', default=64, help='Bin size for histogram')
    parser.add_argument('--save_path', default='/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/output/test', help='save_path for results')
    args = parser.parse_args()
    
    mode = args.mode
    image_path = args.image
    folder_path = args.image_folder
    save_flag = args.save_intermediate
    dict_size = args.dict_size
    histogram_bin = args.histogram_bin
    save_path = args.save_path
    
    if mode == 'kmeans':
    
        first_image = True
        kmeans = None
        
        if image_path is not None and save_path is not None:
            # Save features
            filename = save_path + '_feat.pkl'
            result = get_hist_from_image(image_path, save_path)
            
            # K-Means Part
        
            filename = save_path + '_kmeans.pkl'
            kmeans = construct_kmeans(result)
        
            pickle.dump(kmeans, open(filename, 'wb'))
            
        elif folder_path is not None and save_path is not None:
            # Feature computation and K-Means clustering in batch
            path = folder_path + '*.%s'
            im_list = [glob.glob(path % ext) for ext in ["jpg","png","tif"]]
            for im_l in im_list:
                if len(im_l) != 0:
                    for im_p in im_l:
                        # get filename without extension
                        path_noextend = os.path.splitext(im_p)[0]
                        save_path = im_p + '_feat.pkl'
                        result = get_hist_from_image(image_path, save_path)
                        
                        # Online-Kmeans
                        if first_image:
                            kmeans = construct_kmeans(result)
                        else:
                            partial_fit_k_means(feat, kmeans)
                        first_image = False
            
            filename = save_path + '/kmeans.pkl'        
            pickle.dump(kmeans, open(filename, 'wb'))
        else:
            print('Error: Input image path is None or save path is None.')
        
    #elif mode == 'classifier-train':
        
    
    
def get_hist_from_image(image_path, save_path=None):
    image = cv2.imread(image_path)
    image = np.array(image, dtype=int)
    words = Word(image)
    reslt = np.zeros([words.length, 320])
    
    for word, i in words:
        feat = calculate_feature(word, idx=i, save=save_flag, path=save_path)
        hist = get_histogram(feat, nbins=histogram_bin)
        result[i,:] = hist
    if save_path is not None:
        pickle.dump(result, open(filename, 'wb'))
    return result

    