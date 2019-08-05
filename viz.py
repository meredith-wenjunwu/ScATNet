#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:37:43 2019

@author: wuwenjun
"""

import cv2
from util import *
from cluster import *
from feature import *
from word import Word
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns
sns.set()
import glob


#x = cv2.imread('/projects/medical4/ximing/DistractorProject/page5/A1461_201109151951.jpg')
#x = np.array(x, dtype=int)

# %%
#h, w, _ = x.shape
#width =  10 + 120*5
#x = x[int(0.5*h):int(0.5*h) +width, int(0.5*w):int(0.5*) + width, :]

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


im,_ = load_mat('/projects/medical4/ximing/DistractorProject/page3/roi/2428_1.mat')
loaded_feat = get_feat_from_image(None, False, 120, image=im)

#or word, i in words:
#
#    v2.imwrite('/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/output/test' + str(i) + '.jpg', word)
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
kmeans_filename = '/projects/medical4/ximing/DistractorProject/feature_page3/kmeans.pkl'
feat_filename = '/projects/medical4/ximing/DistractorProject/feature_page3/2428_1_feat.pkl'
hcluster_filename = '/projects/medical4/ximing/DistractorProject/feature_page3/hcluster.pkl'

loaded_kmeans = pickle.load(open(kmeans_filename, 'rb'))
#loaded_feat = pickle.load(open(feat_filename, 'rb'))
loaded_hcluster = pickle.load(open(hcluster_filename, 'rb'))

result = predict_kmeans(loaded_feat, loaded_kmeans, h_cluster=loaded_hcluster)

distances = loaded_kmeans.transform(loaded_feat)
labels = np.unique(result)
print(labels)
words = Word(im)

for l in labels:
    p = os.path.join('/projects/medical4/ximing/DistractorProject/visualize/cluster_page3/', str(l))
    if not os.path.exists(p):
        os.mkdir(p)
#
for idx, r in enumerate(result):
    w = words[idx][0]
    p = os.path.join('/projects/medical4/ximing/DistractorProject/visualize/cluster_page3/', str(r),  '{}_'+ str(idx) + '.jpg')
    cv2.imwrite(p, w)

#Generating summarization pictures

#to_plot = np.zeros([40, 9])
#for l in labels:
#   indices = [i for i, x in enumerate(result) if x == l]
#   dist = [distances[i, l] for i in indices]
#   dist = np.array(dist)
#   ind = np.argsort(dist)
#   print(len(ind))
#   if len(ind) >= 9:
#       to_plot[l,0:9] = ind[0:9]
#   else:
#       to_plot[l,0:len(ind)] = ind[:]

for l in range(40):
    p = os.path.join('/projects/medical4/ximing/DistractorProject/visualize/cluster_page3/', str(l))
    out = os.path.join('/projects/medical4/ximing/DistractorProject/visualize/cluster_page3/', '{}.jpg'.format(l))
    f_ls = glob.glob(os.path.join(p, '*_*.jpg'))
    if len(f_ls) > 9:
        #plot_25 = to_plot[l,:]
        plot_25 = np.random.choice(f_ls, 9)
        #width =  10 + 120*5
        viz = np.ones([3*120+40, 3*120+40, 3])*255
        for idx, f in enumerate(plot_25):
            im = np.array(cv2.imread(f))
            #if (idx <10): print(idx)
#            print('{}:{}, {}:{}'.format(int(idx/3)*120,(int(idx/3)+1)*120, int(idx%3)*120, (int(idx%3)+1)*120))
            #if im.shape[0] > 120:
            viz[int(idx/3)*130:int(idx/3)*130+120, int(idx%3)*130:int(idx%3)*130+120,:] = im

        cv2.imwrite(out, viz)

#plt.cla(); plt.clf()
#plt.hist(labels, bins=np.max(abels)+1, alpha=0.3)
#
#
#plt.ylim([0, 25])
#
#path = '/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/test_hist.jpg'
#plt.savefig(path)

#
# Generating cluster overlay
#kmeans_filename = '/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/output/test_kmeans.pkl'
#loaded_kmeans = pickle.load(open(means_filename, 'rb'))

#result = np.zeros([words.length, 320])

#for word, i in words:
#    feat = calculate_feature(word, idx=i, save=save_flag, path=save_path)
#    hist = get_histogram(feat, nbins=histogram_bin)
 #   result[i,:] = hist

#labels = oaded_kmeans.predict(esult)
#
#uniq = np.unique(labels)
#
#c = sns.color_palette("hls", len(uniq))
#
#
#for word, i in words:
#    b_box = words.bound_box(i)
#    ind, = np.where(uniq==labels[i])
#    x[b_box[0]:b_box[1], b_box[2]:b_box[3],:] = [channel*255 for channel in c[nd[0]]]
#
#
#border = 10
#
#for i in range(0, x.shape[0], 120):
#    x[i:i+border, :, : ] = 0
#
#for j in range(0, x.shape[1], 120):
#
#    x[:, j:j+border, :] = 0
#
#p = '/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/overlay.jpg'
#cv2.imwrite(, x)
#
#
#plt.cla(); plt.clf()
#plt.hist(labels, bins=np.max(abels)+1, alpha=0.3)
#
#
#plt.ylim([0, 25])
#
#path = '/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/test_hist.jpg'
#plt.savefig(path)
#
#
#





