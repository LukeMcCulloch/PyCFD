#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 07:43:40 2020

@author: lukemcculloch
"""
import numpy as np

class Overload(object):
    
    def __init__(self, vector):
        self.vector = vector
        self.parentcell = []
        
    
    def __add__(self, other):
        if isinstance(other, Overload):
            return self.vector + other.vector
        else:
            return self.vector + other
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Overload):
            return self.vector - other.vector
        else:
            return self.vector - other
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    