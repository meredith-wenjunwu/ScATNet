#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 13:02:12 2019

@author: wuwenjun
"""

import math
import cv2

class Word:
    """
    Iterator Class for constructing the word (120x120 patch)
    
    """
    def __init__(self, img):
        """
        Initializer for the word class
        
        Args:
            img: the input image (NxMx3) in numpy array
        
        """
        h,w,_ = img.shape
        h_pad = 120 - h%120
        w_pad = 120 - w%120 
        top = math.floor(h_pad/2)
        bottom = math.ceil(h_pad/2)
        left = math.floor(w_pad/2)
        right = math.ceil(w_pad/2)
    
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
        h,w,_ = padded.shape
        assert h%120 == 0, "height after padding is not divisible by 120"
        assert w%120 == 0, "width after padding is not divisible by 120"
        self.img = padded
        self.h = h
        self.w = w
        self.length = int((self.h/120) * (self.w/120))
        
    def __len__(self):
        """
        Function that return the length of the words/number of 
        word in the image  
    
        """
        return (self.h/120) * (self.w/120)
        
    def bound_box(self, idx):
        """
        Function that return the bounding box of a word given its index
        Args:
            ind: int, ind < number of words
        
        Returns:
            Bounding box(int[]): [h_low, h_high, w_low, w_high]
        """
        assert idx < self.length, "Index Out of Bound"
        num_word_w = int(self.w/120)
        h = math.floor(idx / num_word_w) * 120
        w = int(idx % (num_word_w) * 120)
        
        return [h, h+120, w, w+120]
        
    
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
        
