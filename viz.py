#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:37:43 2019

@author: wuwenjun
"""

import cv2
from feature import *
from word import Word
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns
sns.set()



x = cv2.imread('/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/test.jpg')
x = np.array(x, dtype=int)

# %%
h, w, _ = x.shape
width =  10 + 120*5
x = x[int(0.5*h):int(0.5*h) +width, int(0.5*w):int(0.5*w) + width, :]

## %%
#border = 10
#
#for i in range(0, x.shape[0], 120):
#    x[i:i+border, :, : ] = 0
#    

#for j in range(0, x.shape[1], 120):
#    x[:, j:j+border, :] = 0
#
#   



words = Word(x)

#for word, i in words:
#    
#    cv2.imwrite('/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/output/test' + str(i) + '.jpg', word)
#    feat = calculate_feature(word, idx=i, save=True, path='/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/output/test')
#    plt.cla(); plt.clf()
#    plt.hist(x[:,:,0].reshape(-1), color="red", bins=64, alpha=0.3)
#    plt.hist(x[:,:,1].reshape(-1), color="green", bins=64, alpha=0.3)
#    plt.hist(x[:,:,2].reshape(-1), color="blue", bins=64, alpha=0.3)
#    
#    plt.ylim([0, 1000])
#
#    path = '/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/output/test' + str(i) + '_hist.jpg'
#    plt.savefig(path)


# Generating cluster images
#kmeans_filename = '/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/output/test_kmeans.pkl'
#feat_filename = '/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/output/test_feat.pkl'
#
#loaded_kmeans = pickle.load(open(kmeans_filename, 'rb'))
#loaded_feat = pickle.load(open(feat_filename, 'rb'))
#
#result = loaded_kmeans.predict(loaded_feat)
#
#labels = np.unique(result)
#
#for l in labels:
#    p = os.path.join('/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/output', str(l))
#    if not os.path.exists(p):
#        os.mkdir(p)
#    
#for idx, r in enumerate(result):
#    w = words[idx][0]
#    p = os.path.join('/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/output', str(r)) + '/'+ str(idx) + '.jpg'
#    cv2.imwrite(p, w)
    
    
# Generating cluster overlay
kmeans_filename = '/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/output/test_kmeans.pkl'
loaded_kmeans = pickle.load(open(kmeans_filename, 'rb'))

result = np.zeros([words.length, 320])
        
for word, i in words:
    feat = calculate_feature(word, idx=i, save=save_flag, path=save_path)
    hist = get_histogram(feat, nbins=histogram_bin)
    result[i,:] = hist

labels = loaded_kmeans.predict(result)

uniq = np.unique(labels)

c = sns.color_palette("hls", len(uniq))



  
for word, i in words:
    b_box = words.bound_box(i)
    ind, = np.where(uniq==labels[i])
    x[b_box[0]:b_box[1], b_box[2]:b_box[3],:] = [channel*255 for channel in c[ind[0]]]

# %%
border = 10

for i in range(0, x.shape[0], 120):
    x[i:i+border, :, : ] = 0
    
for j in range(0, x.shape[1], 120):
    
    x[:, j:j+border, :] = 0

p = '/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/overlay.jpg'
cv2.imwrite(p, x) 


plt.cla(); plt.clf()
plt.hist(labels, bins=np.max(labels)+1, alpha=0.3)


plt.ylim([0, 25])

path = '/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/test_hist.jpg'
plt.savefig(path) 

      






