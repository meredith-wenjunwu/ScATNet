#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 00:56:55 2019

@author: wuwenjun
"""
from sklearn.cluster import MIniBatchKMeans


def construct_kmeans(feat, dict_size=40):
    """function that calculate the kmeans cluster from feature vector 
    with given dictionary size.

    Args:
        feat: the input feature (320xN) in numpy array
        dict_size: the number of clusters for kmeans

    Returns:
        The resulted kmeans cluster model

    """
    kmeans = MiniBatchKMeans(n_clusters=dict_size, random_state=0).partial_fit(feat)
    
    return kmeans

def partial_fit_k_means(feat, kmeans=None):
    if kmeans is None:
        return construct_kmeans(feat)
    return kmeans.partial_fit(feat)

def predict_kmeans(feat, kmeans):
    """function that assign word to closest k-means cluster.

    Args:
        features: the input feature (320xN) in numpy array
        kmeans: the kmeans cluster model
        
    Returns:
        (array) The resulted cluster each sample belongs to
    """
    return kmeans.predict(feat)