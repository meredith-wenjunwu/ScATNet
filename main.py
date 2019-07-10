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
from cluster import *
import os
import pickle
import glob
from util import *




if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--mode',
                        default='kmeans',
                        const='kmeans',
                        nargs='?',
                        choices=['feature', 'kmeans', 'kmeans_visual', 'classifier_train', 'classifier_test'],
                        help='Choose mode from k-means clustering, visualization and classification_training, classification_testing')
    parser.add_argument('--single_image', default=None, help='Input image path')
    parser.add_argument('--image_folder', default=None, help='Input image batch folder' )
    parser.add_argument('--save_intermediate', default=False, help='Whether or not to save the intermediate results')
    parser.add_argument('--dict_size', default= 40, help='Dictionary Size for KMeans')
    parser.add_argument('--histogram_bin', default=64, help='Bin size for histogram')
    parser.add_argument('--save_path', default='/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/output/test', help='save_path for outputs: e.g.features, kmeans, classifier')
    parser.add_argument('--word_size', default=120, help='Size of a word (in a bag of words model)')
    parser.add_argument('--bag_size', default=3600, help="Size of a bag (in a bag of words model)')
    parser.add_argument('--overlap_bag', default=2400, help='Overlapping pixels between bags')
    parser.add_argument('--classifier', default='logistic',
                        const='logistic',
                        nargs='?',
                        choices=['logistic', 'svm'])
    args = parser.parse_args()

    mode = args.mode
    image_path = args.single_image
    folder_path = args.image_folder
    save_flag = args.save_intermediate
    dict_size = args.dict_size
    histogram_bin = args.histogram_bin
    save_path = args.save_path
    word_size = args.word_size
    bag_size = args.bag_size
    overlap = args.overlap_bag

    if mode == 'kmeans':

        first_image = True
        kmeans = None

        if image_path is not None and save_path is not None:
            # Save features
            filename = save_path + '_feat_word.pkl'
            result = get_feat_from_image(image_path, save_flag, word_size, fiename)

            # K-Means Part

            filename = save_path + '_kmeans.pkl'
            kmeans = construct_kmeans(result)

            pickle.dump(kmeans, open(filename, 'wb'))

        elif folder_path is not None and save_path is not None:
            print('-------Running Batch Job-------')
            # Feature computation and K-Means clustering in batch
            im_list = [glob.glob(path % ext) for ext in ["jpg","png","tif"]]
            for im_l in im_list:
                if len(im_l) != 0:
                    print('# of images: %r' %(len(im_l)))
                    count = 0
                    for im_p in im_l:
                        if count % 10 == 0: print('Processed %r / %r' %(count, len(im_l)))
                        count += 1
                        # get filename without extension
                        base = os.path.basename(im_p)
                        path_noextend = os.path.splitext(base)[0]
                        fname = path_noextend  + '_feat.pkl'
                        filename = os.path.join(filename, fname)
                        result = get_feat_from_image(im_p, save_flag, word_size, filename)

                        # Online-Kmeans
                        if first_image:
                            kmeans = construct_kmeans(result)
                        else:
                            partial_fit_k_means(result, kmeans)
                        first_image = False
                        assert kmeans is not None, "kmeans construction/update invalid"
            if kmeans is not None:
                filename = os.path.join(save_path, 'kmeans.pkl')
                pickle.dump(kmeans, open(filename, 'wb'))
            
            #Compute histogram from 
        else:
            print('Error: Input image path is None or save path is None.')

    elif mode == 'classifier-train':
        filename = os.path.join(save_path, 'kmeans.pkl')
        loaded_kmeans = pickle.load(open(kmeans_filename, 'rb'))
        assert folder_path is not None or image_path is not None, "Error: Input image path is None or save path is None."
        assert save_path is not None, "Save Path is None"
        assert loaded_kmeans is not None, "Path incorrect/File doesnt exist"
        
        if image_path is not None:
            filename = save_path + '_feat_bag.pkl'
            result = get_hist_from_image(image_path, loaded_kmeans, dict_size, word_size, 
                            bag_size, overlap, save_flag, save_path)
            
            
            
            
            
        
        
        
        



