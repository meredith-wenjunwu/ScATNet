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

    def padding(self, img, window_size, h=None, w=None):
        if img is not None:
            h, w, _ = img.shape
        assert h is not None
        assert w is not None
        if h % window_size != 0:
            h_pad = window_size - h % window_size
        else:
            h_pad = 0
        if w % window_size != 0:
            w_pad = window_size - w % window_size
        else:
            w_pad = 0
        self.top = math.floor(h_pad / 2)
        self.bottom = math.ceil(h_pad / 2)
        self.left = math.floor(w_pad / 2)
        self.right = math.ceil(w_pad / 2)
        if img is not None:
            padded = cv2.copyMakeBorder(
                img, self.top, self.bottom, self.left, self.right,
                cv2.BORDER_REFLECT)
            self.h, self.w, _ = padded.shape
            return padded
        self.h = h + self.top + self.bottom
        self.w = w + self.left + self.right
        return img

    def __init__(self, img=None, h=None, w=None, size=3600, overlap_pixel=2400,
        padded=True):
        """
        Initializer for the bag class

        Args:
            img: the input image (NxMx3) in numpy array

        """
        # h,w,_ = img.shape
        # h_pad = size - h%size
        # w_pad = size - w%size
        # top = math.floor(h_pad/2)
        # bottom = math.ceil(h_pad/2)
        # left = math.floor(w_pad/2)
        # right = math.ceil(w_pad/2)

        # padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
        # h,w,_ = padded.shape
        self.h = None
        self.w = None
        if padded:
            img = self.padding(img, size, h, w)
            assert self.h % size == 0, "height after padding is not divisible by 3600"
            assert self.w % size == 0, "width after padding is not divisible by 3600"
        if img is not None:
            h, w, _ = img.shape
            self.h = h
            self.w = w
        else:
            if not padded:
                assert h is not None, "Need to provide height when image is absent"
                assert w is not None, "Need to provide width when image is absent"
                self.h = h
                self.w = w
        self.img = img
        self.overlap_pixel = overlap_pixel
        self.size = size
        self.length = math.floor(max(math.floor((self.h - self.overlap_pixel) /
            (self.size - self.overlap_pixel)), 0) * max(math.floor((self.w -
            self.overlap_pixel) / (self.size - self.overlap_pixel)), 0))


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
        num_bag_w = math.floor((self.w - self.overlap_pixel) /
                      (self.size - self.overlap_pixel))
        h = math.floor(math.floor(idx / num_bag_w) * (self.size -
           self.overlap_pixel))
        w = math.floor(idx % (num_bag_w) * (self.size - self.overlap_pixel))

        return [h, h + self.size, w, w + self.size]

    def __getitem__(self, idx):
        """
        Function that returns the word at a index
        Args:
            idx: int, ind < number of words

        """
        assert self.img is not None, "Try to get item image is absent"
        if idx >= self.length:
            raise StopIteration

        b_box=self.bound_box(idx)

        return [self.img[b_box[0]:b_box[1], b_box[2]:b_box[3], :], idx]
