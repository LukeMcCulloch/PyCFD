#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:55:49 2019

@author: luke
"""
import numpy as np


def Grad2D(u):
    """
    2D gradient of a scalar
    
    basis function style:
    """
    ur = Dr*u
    us = Ds*u
    ux = np.multiply(rx,ur) + \ 
            np.multiply(sx,us) 
    uy = np.multiply(ry,ur) + \ 
            np.multiply(sy,us) 
    return ux,uy
    
    