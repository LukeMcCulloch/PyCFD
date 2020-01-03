# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:35:53 2020

@author: Luke.McCulloch
"""

import numpy as np

class Node(object):
    def __init__(self, vector):
        self.x0 = vector[0]
        self.x1 = vector[1]
        self.vector = vector
        
        
class Cell(object):
    def __init__(self, nodes):
        self.nodes = nodes


class Grid(object):
    
    def __init__(self, mesh=None, m=10,n=10):
        if mesh is None:
            mesh = np.zeros((2,m,n),float) # C-ordering.  last index is most rapid
            self.dim = 2
            self.m = m
            self.n = n
        else:
            shp = np.shape(mesh)
            self.dim = shp[0]
            self.m = shp[1]
            self.n = shp[2]
            
        mms = np.linspace(0.,1.,m)
        nms = np.linspace(0.,1.,n)
        self.mesh = mesh
        #
        self.nodes = []
        for i in range(self.m):
            self.nodes.append([])
            for j in  range(self.n):
                self.mesh[0,i,j] = mms[i]
                self.mesh[1,i,j] = mms[j]
                self.nodes[i].append(Node(self.mesh[:,i,j]))
                
        self.nodes = np.asarray(self.nodes)
                
        
        
if __name__ == '__main__':
    self = Grid()