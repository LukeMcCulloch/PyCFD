#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:04:47 2020

@author: lukemcculloch
"""

import numpy as np
# imports for plotting:
import matplotlib.pyplot as plt

class PlotGrid(object):
    
    def __init__(self, grid):
        self.grid = grid
        
    
    def plot_cells(self, canvas = None,
                   alpha=.1):
        if canvas is None:
            fig, ax = plt.subplots()
            ax.axis('equal')
        else:
            ax = canvas
        grid = self.grid
        
        #for cell in grid.cellList[:1]:
        for cell in grid.cellList:
            for face in cell.faces:
                n0 = face.nodes[0]
                n1 = face.nodes[1]
                
                ax.plot(n0.x0,n0.x1,
                        color='red',
                        marker='o',
                        alpha = alpha)
                ax.plot(n1.x0,n1.x1,
                        color='red',
                        marker='o',
                        alpha = alpha)
                
                x = [n0.x0,n1.x0]
                y = [n0.x1,n1.x1]
                ax.plot(x, y,
                        color='black',
                        alpha = alpha)
        
        return ax
    
    
    def plot_centroids(self, canvas = None,
                       alpha=.1):
        if canvas is None:
            fig, ax = plt.subplots()
        else:
            ax = canvas
        grid = self.grid
        
        
        #for cell in grid.cellList[:1]:
        for cell in grid.cellList:
            ax.plot(cell.centroid[0],
                    cell.centroid[1],
                    color='green',
                    marker='o',
                    alpha = alpha,)
        
        return ax
    
    
    def plot_face_centers(self, canvas = None,
                       alpha=.1):
        if canvas is None:
            fig, ax = plt.subplots()
        else:
            ax = canvas
        grid = self.grid
        
        
        #for cell in grid.cellList[:1]:
        for cell in grid.cellList:
            for face in cell.faces:
                
                norm0 = face.normal_vector - face.center
                norm1 = face.normal_vector - face.center
                
                ax.plot(face.center[0],
                        face.center[1],
                        color='yellow',
                        marker='o',
                        alpha = alpha)
                
                ax.plot(face.center[0],
                        face.center[1],
                        color='yellow',
                        marker='o',
                        alpha = alpha)
        
        return ax
    
    def plot_normals(self, canvas = None,
                       alpha=.4):
        """
        debugging:
            ax = axTri
        """
        if canvas is None:
            fig, ax = plt.subplots()
        else:
            ax = canvas
        grid = self.grid
        
        
        #for cell in grid.cellList[:1]:
        for cell in grid.cellList:
            for face in cell.faces:
                # print 'new face'
                # print '\n Normal vector'
                # print face.normal_vector
                # print '\n center'
                # print face.center
                
                fnorm = face.normal_vector
                #norm0 = .5*face.normal_vector*face.area**2 + face.center
                #norm0 = norm0*face.area
                norm = 2.*np.linalg.norm(face.normal_vector)*face.area
                #ax.plot([ norm0[0],face.center[0] ],
                #        [ norm0[1],face.center[1] ],
                #        color='purple',
                #        marker='o',
                #        alpha = alpha)
                
                #scalearrow = np.linalg.norm(norm0)
                #dx =  (fnorm[0]-face.center[0])/norm
                #dy =  (fnorm[1]-face.center[1])/norm
                
                plt.arrow(x=face.center[0],
                          y=face.center[1],
                          dx=fnorm[0]/norm ,
                          dy=fnorm[1]/norm )
                
                # bad!
                # plt.arrow(x=face.center[0],
                #           y=face.center[1],
                #           dx=dx ,
                #           dy=dy )
        return ax
    
    def plot_cell(self, cell, canvas = None,
                  alpha=.4):
        if canvas is None:
            fig, ax = plt.subplots()
        else:
            ax = canvas
        grid = self.grid
        
        
        #for cell in grid.cellList[:1]:
        for cell in grid.cellList:
            for face in cell.faces:
                
                norm0 = face.normal_vector + face.center
                
                ax.plot([ norm0[0],face.center[0] ],
                        [ norm0[1],face.center[1] ],
                        color='black',
                        marker='o',
                        alpha = alpha)
                
        return ax
    
    def plot_boundary(self, canvas = None, alpha=.1):
        '''
        plot boundary 
        '''
        if canvas is None:
            fig, ax = plt.subplots()
            ax.axis('equal')
        else:
            ax = canvas
        grid = self.grid
        
        for seg in grid.bound:
            self.plot_bgrid(seg, ax, alpha)
        
        for face in grid.boundaryList:
            ax = face.plot_face_normal(ax)
            
        return ax
    
    def plot_bgrid(self, seg, canvas = None, alpha=.1):
        '''
        plot boundary segment
        '''
        if canvas is None:
            fig, ax = plt.subplots()
            ax.axis('equal')
        else:
            ax = canvas
        grid = self.grid
        
        for i in range(1,seg.nbnodes-1):
            n0 = grid.nodes[seg.bnode[i-1]]
            n1 = grid.nodes[seg.bnode[i]]
            
            
            
            ax.plot(n0.x0,n0.x1,
                    color='red',
                    marker='o',
                    alpha = alpha)
            ax.plot(n1.x0,n1.x1,
                    color='red',
                    marker='o',
                    alpha = alpha)
            
            x = [n0.x0,n1.x0]
            y = [n0.x1,n1.x1]
            ax.plot(x, y,
                    color='black',
                    alpha = alpha)
        
        return ax