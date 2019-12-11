import numpy as np
import scipy.misc
from skimage import feature
from multiprocessing.pool import ThreadPool
from skimage.color import rgb2gray, rgb2lab
import os
import cv2
from normalizeStaining import normalizeStaining


def calculate_HE(img, idx=None, save=False, path=None):
    """function that calculates the HE image from an input image.

    Args:
        img: the input image (NxMx3) in numpy array
        save: flag to indicate whether the result need to be saved
        idx: index for save path
        path: the path to save the result

    Returns:
        The resulted HE image (NxMx2) in numpy array

    """
    
    M = np.array([ [0.65, 0.70, 0.29],
     [0.07, 0.99, 0.11],
     [0.27, 0.57, 0.78] 
     ]
    )

    # Normalize to have unit length 1
    M = np.divide(M, np.sum(M, axis=1).reshape(3,1))
    M[np.isnan(M)] = 0

    he = np.matmul(img, M.T)


    if save and path and idx is not None:
        saveFile = path + '_' + str(idx)
    else:
        saveFile = None
    
    Inorm, H, E = normalizeStaining(img, saveFile=saveFile)

    return np.concatenate((H, E), axis=2)


def calculate_LBP(img):
    """function that calculates LBP features from an input image.

    Args:
        img: the input HE image (NxMx2) or rgb image (NxMx3) in numpy array
        nbins: the number of bins for the histogram
        save: flag to indicate whether the result need to be saved
        idx: index for save path
        path: the path to save the result

    Returns:
        The resulted LBP image (NxMx2 or NxMx1) in numpy array

    """
    
    h, w, d = img.shape

    output = np.zeros([h, w, d])
    
    for i in range(d):
        output[:,:,i] = feature.local_binary_pattern(img[:,:, i], 8, 1)

    return output
    
def calculate_feature(img, idx=None, save=False, path=None):
    """function that calculates features from an input image.

    Args:
        img: the input rgb image (NxMx3) in numpy array
        save: flag to indicate whether the result need to be saved
        idx: index for save path
        path: the path to save the result

    Returns:
        The resulted feature vector (NxMx5) in numpy array

    """

    HE = calculate_HE(img, idx, save, path)
    h, w, d = img.shape

    lbp = calculate_LBP(HE)

    lab = rgb2lab(img)

    all_features = np.concatenate([lbp, lab], axis=2)
    return all_features

def get_histogram(features, nbins=64):
    """function that calculates features from an input image.

    Args:
        features: the input feature (NxMx5) in numpy array
        nbins: the number of bins for the histogram

    Returns:
        The resulted feature vector (320x1) in numpy array

    """
    _,_, d = features.shape
    result = []
    
    for i in range(d):
        r = (np.min(features[:, :, i]), np.max(features[:, :, i]))
        hist, edges_ = np.histogram(features[:,:, i], range=r, bins=nbins)
        hist = hist / np.sum(hist) # normalize it to have the percetage instead of count
        result += hist.reshape(-1).tolist()

    result = np.array(result).reshape(-1)
    assert len(result) == nbins * d, "Histogram feature length is not right."
    
    
    return result

def get_histogram_cluster(cluster_words, dict_size=40):
    hist, edges_ = np.histogram(cluster_words, range=(0, 40), bins=dict_size)
    return hist