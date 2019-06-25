#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 00:31:50 2019

@author: wuwenjun
"""

from word import Word

class Bag:
    """
    Iterator Class for constructing the bag of words (3600x3600 patch)
    
    """
    
    def __init__(self, img):
        """
        Initializer for the bag class
        
        Args:
            img: the input image (NxMx3) in numpy array
        
        """
    