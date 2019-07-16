#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 00:31:50 2019

@author: wuwenjun
"""

import math
import cv2

class Bag:
    """
    Iterator Class for constructing the bag of words (3600x3600 patch)

    """

    def padding(self, img, window_size):
        h,w,_ = img.shape
        h_pad = window_size - h%window_size
        w_pad = window_size - w%window_size
        top = math.floor(h_pad/2)
        bottom = math.ceil(h_pad/2)
        left = math.floor(w_pad/2)
        right = math.ceil(w_pad/2)

        padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
        return padded

    def __init__(self, img, label=None, size=3600, overlap_pixel=2400):
        """
        Initializer for the bag class

        Args:
            img: the input image (NxMx3) in numpy array

        """
        #h,w,_ = img.shape
        #h_pad = size - h%size
        #w_pad = size - w%size
        #top = math.floor(h_pad/2)
        #bottom = math.ceil(h_pad/2)
        #left = math.floor(w_pad/2)
        #right = math.ceil(w_pad/2)

        #padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
        #h,w,_ = padded.shape

        padded = self.padding(img, size)
        h,w,_ = padded.shape

        assert h%size == 0, "height after padding is not divisible by 3600"
        assert w%size == 0, "width after padding is not divisible by 3600"
        self.img = padded
        self.h = h
        self.w = w
        self.overlap_pixel = overlap_pixel
        self.size = size
        self.length = int((self.h-self.overlap_pixel)/(self.size-self.overlap_pixel) * (self.w-self.overlap_pixel)/(self.size-self.overlap_pixel))
        self.label = label


    def __len__(self):
        """
        Function that return the length of the words/number of
        word in the image

        """
        return self.length


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

    def __getitem__(self, idx):
        """
        Function that returns the word at a index
        Args:
            idx: int, ind < number of words

        """

        if idx >=self.length:
            raise StopIteration

        b_box = self.bound_box(idx)

        return [self.img[b_box[0]:b_box[1], b_box[2]:b_box[3], :], idx]




