# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:35:53 2020

@author: Luke.McCulloch
"""

import numpy as np

from Overload import Overload

from Utilities import normalize, normalized, norm, dot, cross, \
    normalize2D, normalized2D

class Node(Overload):
    def __init__(self, vector):
        self.x0 = vector[0]
        self.x1 = vector[1]
        self.vector = vector
        
    #def __call__(self):
    #    return self.vector

class Face(object):
    '''
    In 2D, a face is an edge
    '''
    def __init__(self, nodes):
        self.nodes = nodes
        self.N = 2 #len(nodes)
        self.center = np.zeros((self.N),float)
        #sumx0 = sum( [el.x0 for el in self.nodes] )
        #sumx1 = sum( [el.x1 for el in self.nodes] )
        self.area = np.linalg.norm(self.nodes[1]-self.nodes[0])
            
    def normal(self):
        """
        cross vector with 0 in x2 dir
         with vector with 1 in the x2 dir
         
         this is of course (x1,-x0, 0)
         
         normals point in
         
        """
        vec = self.nodes[1] - self.nodes[0]
        dumvec1 = np.zeros((3),float)
        dumvec2 = np.zeros_like((dumvec1))
        dumvec1[:-1] = self.nodes[1] - self.nodes[0]
        dumvec2[:-1] = self.nodes[1] - self.nodes[0]
        dumvec2[-1] = 1.
        
        
        n3 = normalize(cross(dumvec1,dumvec2) )
        return n3[:-1]
                      
        
        
class Cell(object):
    """
    The Cij'th cell
    
    ccw-winding
    """
    def __init__(self, nodes):
        self.nodes = nodes
        self.N = len(self.nodes)
        self.num_faces = self.N
        self.F = np.asarray((self.num_faces),float)
        self.G = np.asarray((self.num_faces),float)
        self.faces = []
        self.set_face_vectors()
        
    def set_face_vectors(self):
        for i in range(self.N):
            self.faces.append( Face( [self.nodes[i],
                                      self.nodes[(i+1)%self.N] 
                                      ] ))
        return
    
    def get(self, i):
        assert(i<=self.N),'error, i>N'
        return self.N%i
    
    def face(self, i):
        return 
    
    def print_vertices(self):
        for vert in self.nodes:
            print vert.vector
        return
    
    
    def print_faces(self, normals=False):
        for face in self.faces:
            print 'face'
            for vert in face.nodes:
                print vert.vector
            if normals:
                print 'face normal'
                print face.normal()
        return


class Grid(object):
    
    def __init__(self, mesh=None, m=10,n=10,type_='rect'):
        self.gridtype = {'rect':0,
                         'tri':1}
        self.type = type_
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
                self.mesh[1,i,j] = nms[j]
                self.nodes[i].append(Node(self.mesh[:,i,j]))
                
        self.nodes = np.asarray(self.nodes)
        
        self.make_cells()
        
    def make_cells(self):
        if self.type is 'rect':
            self.make_rect_cells()
        else:
            self.make_tri_cells()
        return
    
    def make_rect_cells(self):
        
        self.cells = []
        for i in range(self.m-1):
            self.cells.append([])
            for j in  range(self.n-1):
                
                self.cells[i].append(
                                    Cell([
                                            self.nodes[i,j],
                                            self.nodes[i,j+1],
                                            self.nodes[i+1,j+1],
                                            self.nodes[i+1,j]
                                        ])  )
                
        self.cells = np.asarray(self.cells)
        return
    
    def make_tri_cells(self):
        
        self.cells = []
        for i in range(self.m-1):
            self.cells.append([])
            for j in  range(self.n-1):
                
                self.cells[i].append(
                                    Cell([
                                            self.nodes[i,j],
                                            self.nodes[i,j+1],
                                            self.nodes[i+1,j]
                                        ]) )
                
                self.cells[i].append(
                                    Cell([
                                            self.nodes[i,j+1],
                                            self.nodes[i+1,j+1],
                                            self.nodes[i+1,j]
                                        ])
                )
                
        self.cells = np.asarray(self.cells)
        return
        

        
if __name__ == '__main__':
    gd = Grid(type_='rect')
    self = Grid(type_='tri')