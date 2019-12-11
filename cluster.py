#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 00:56:55 2019

@author: wuwenjun
"""
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering

def construct_kmeans(feat, init_size=200):
    """
    function that calculate the kmeans cluster from feature vector
    with given dictionary size.

    Args:
        feat: the input feature (320xN) in numpy array
        dict_size: the number of clusters for kmeans

    Returns:
        The resulted kmeans cluster model

    """
    kmeans = MiniBatchKMeans(n_clusters=init_size, random_state=0).partial_fit(feat)
    assert kmeans is not None
    return kmeans

def partial_fit_k_means(feat, kmeans=None):
    if kmeans is None:
        return construct_kmeans(feat)
    return kmeans.partial_fit(feat)

def h_cluster(kmeans, final_size=40, affinity='euclidean'):
    """
    function that H-Clusters kmeans

    Args:
        kmeans: existing kmeans cluster model

    Returns:
        hirachical model
    """
    assert kmeans is not None, "Invalid input: None"
    Kx = kmeans.cluster_centers_
    Hclustering = AgglomerativeClustering(n_clusters=40,
             affinity=affinity)
    Kx_mapping = {case:cluster for case,
             cluster in enumerate(kmeans.labels_)}
    Hclustering.fit(Kx)
    return Hclustering


def predict_kmeans(feat, kmeans, h_cluster=None):
    """
    function that assign word to closest k-means cluster.

    Args:
        features: the input feature (320xN) in numpy array
        kmeans: the kmeans cluster model

    Returns:
        (array) The resulted cluster each sample belongs to
    """
    if h_cluster is None:
        return kmeans.predict(feat)
    Kx = kmeans.cluster_centers_
    K_mapping = kmeans.predict(feat)
    H_mapping = h_cluster.fit_predict(Kx)
    #print(H_mapping)
    return [H_mapping[cluster] for cluster in K_mapping]

