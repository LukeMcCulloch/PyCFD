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
        self.u = np.zeros((mesh.nCells,nq),float) # conservative variables at cells/nodes
        self.w = np.zeros((mesh.nCells,nq),float) # primative variables at cells/nodes
        self.gradw = np.zeros((mesh.nCells,nq,2),float) # gradients of w at cells/nodes.
        # 
        # solution convergence
        self.res = np.zeros((len(mesh.cellList),nq),float) #residual vector
        self.residual_norm = np.zeros((nq,1),float)
        
        # update step data
        self.u0 = np.zeros((mesh.nCells,nq),float)
        
        # accessor integers for clarity
        self.ir = 0 # density
        self.iu = 1 # x-velocity
        self.iv = 2 # y-velocity
        self.ip = 3 # pressure
        
        # fluid properties
        self.gamma = 1.4 # Ratio of specific heats for air
        
        
        
        
        
        
        
        
        
        
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
    
    
    
    
    #-------------------------------------------------------------------------#
    #
    # compute residuals
    #
    #-------------------------------------------------------------------------#
    
    #-------------------------------------------------------------------------#
    #
    # compute_residual: comptutes the local residual 
    #
    #-------------------------------------------------------------------------#
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
    
    
    
    #-------------------------------------------------------------------------#
    #
    # time stepping
    #
    #-------------------------------------------------------------------------#
    def compute_global_time_step(self):
        return
    
    
    #-------------------------------------------------------------------------#
    #
    # compute w from u
    # ------------------------------------------------------------------------#
    #  Input:  u = conservative variables (rho, rho*u, rho*v, rho*E)
    # Output:  w =    primitive variables (rho,     u,     v,     p)
    # ------------------------------------------------------------------------#
    #
    # Note:    E = p/(gamma-1)/rho + 0.5*(u^2+v^2)
    #       -> p = (gamma-1)*rho*E-0.5*rho*(u^2+v^2)
    # 
    #
    #-------------------------------------------------------------------------#
    def u2w(self, u):
        w = np.zeros((nq),float)
        
        w(self.ir) = u[0]
        w(self.iu) = u[1]/u[0]
        w(self.iv) = u[2]/u[0]
        w(self.ip) = (self.gamma-1.0)*( u[3] - \
                                       0.5*w[0]*(w[1]*w[1] + w[2]*w[2]) )
        return