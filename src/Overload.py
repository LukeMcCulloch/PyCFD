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
        
    
    def __add__(self, other):
        return self.vector + other.vector
    
    