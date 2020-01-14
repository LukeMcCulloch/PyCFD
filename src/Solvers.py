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
        
        # solution data
        self.u = np.zeros((mesh.nCells,nq),float)
        self.w = np.zeros((mesh.nCells,nq),float)
        self.u2w = np.zeros((mesh.nCells,nq),float)
        # 
        # solution convergence
        self.res = np.zeros((len(mesh.cellList),nq),float)
        self.residual_norm = np.zeros((nq,1),float)
        
        # update step data
        self.u0 = np.zeros((mesh.nCells,nq),float)
        
        
        
        
        
        
        
        
        
        
        
        
    #-------------------------------------------------------------------------#
    # Euler solver: Explicit Unsteady Solver: Ut + Fx + Gy = S
    #
    # This subroutine solves an un steady problem by 2nd-order TVD-RK with a
    # global time step.
    #-------------------------------------------------------------------------#
    def explicit_unsteady_solver(self):
        time = 0.0
        
        while (time < t_final):
            #------------------------------------------------------------------
            # Compute the residual: res(i,:)
            self.compute_residual()
            
            #------------------------------------------------------------------
            # Compute the global time step, dt. One dt for all cells.
            self.compute_global_time_step()
            
            #------------------------------------------------------------------
            # Increment the physical time and exit if the final time is reached
            time += dt
            
            #------------------------------------------------------------------
            # Update the solution by 2nd-order TVD-RK.: u^n is saved as u0(:,:)
            #  1. u^*     = u^n - (dt/vol)*Res(u^n)
            #  2. u^{n+1} = 1/2*(u^n + u^*) - 1/2*(dt/vol)*Res(u^*)
            
            #-----------------------------
            #- 1st Stage of Runge-Kutta:
            
        return
    
    
    def compute_residual_norm(self):
        return
    
    #-------------------------------------------------------------------------#
    #
    # compute_residual: comptutes the residuals at cells for
    # the cell-centered finite-volume discretization.
    #
    #-------------------------------------------------------------------------#
    def compute_residual(self):
        mesh = self.mesh
        for cell in mesh.cellList:
        return
    
    def compute_global_time_step(self):
        return