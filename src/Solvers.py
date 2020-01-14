#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:15:20 2020

@author: lukemcculloch
"""
import numpy as np

nq = 4 # Euler system size

def Solvers(object):
    def __init__(self, mesh):
        self.mesh = mesh
        self.res = np.zeros((len(mesh.cellList),nq),float)
        self.residual_norm = np.zeros((nq,1),float)
        
    #-----------------------------------------------------------------------------#
    # Euler solver: Explicit Unsteady Solver: Ut + Fx + Gy = S
    #
    # This subroutine solves an un steady problem by 2nd-order TVD-RK with a
    # global time step.
    #-----------------------------------------------------------------------------#
    def explicit_steady_solver(self):
        time = 0.0
        
        while (time < t_final):
            #-------------------------------------------------------------------
            # Compute the residual: res(i,:)
            self.compute_residual()
            
        return
    
    
    def compute_residual_norm(self):
        return
    
    #-----------------------------------------------------------------------------#
    #
    # compute_residual: comptutes the residuals at cells for
    # the cell-centered finite-volume discretization.
    #
    #-----------------------------------------------------------------------------#
    def compute_residual(self):
        mesh = self.mesh
        for cell in mesh.cellList:
            
            
        return