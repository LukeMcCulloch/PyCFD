#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:15:20 2020

@author: lukemcculloch
"""
import numpy as np

nq = 4 # Euler system size

class cclsq(object):
    
    def __init__(self, mesh):
        self.mesh = mesh
        self.nNodes = mesh.nNodes
        self.nCells = mesh.nCells
        #
        self.nnghbrs_lsq    #number of lsq neighbors
        self.nghbr_lsq      #list of lsq neighbors
        self.cx             #LSQ coefficient for x-derivative
        self.cy             #LSQ coefficient for y-derivative
        
        self.cclsq = np.zeros()
        
    def construct_vertex_stencil(self):
        return

class Solvers(object):
    def __init__(self, mesh):
        self.mesh = mesh
        
        self.second_order = True
        self.use_limiter = True
        
        # solution data
        self.u = np.zeros((mesh.nCells,nq),float) # conservative variables at cells/nodes
        self.w = np.zeros((mesh.nCells,nq),float) # primative variables at cells/nodes
        self.gradw = np.zeros((mesh.nCells,nq,2),float) # gradients of w at cells/nodes.
        # 
        # solution convergence
        self.res = np.zeros((len(mesh.cellList),nq),float) #residual vector
        self.res_norm = np.zeros((nq,1),float)
        #
        # local convergence storage saved for speed
        self.gradw1 = np.zeros((nq,2),float)
        self.gradw2 = np.zeros((nq,2),float)
        
        # update step data
        self.u0 = np.zeros((mesh.nCells,nq),float)
        
        # accessor integers for clarity
        self.ir = 0 # density
        self.iu = 1 # x-velocity
        self.iv = 2 # y-velocity
        self.ip = 3 # pressure
        
        # fluid properties
        self.gamma = 1.4 # Ratio of specific heats for air
        self.rho_inf = 1.0
        self.u_inf = 1.0
        self.v_inf = 0.0
        self.p_inf = 1./self.gamma
        
        #------------------------------------------
        #>> Cell-centered limiter data
        #------------------------------------------
        self.limiter_beps = 1.0e-14
        self.phi = np.zeros((mesh.nCells),float)
        
        
        
        
        
        
        
        
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
    #
    # compute_residual: comptutes the local residual 
    #
    #-------------------------------------------------------------------------#
    def compute_residual_norm(self):
        self.res_norm[:] = np.sum(np.abs(self.res)) / float(self.mesh.nCells)
        return
    
    #-------------------------------------------------------------------------#
    #
    # compute_residual: comptutes the residuals at cells for
    # the cell-centered finite-volume discretization.
    #
    #-------------------------------------------------------------------------#
    def compute_residual(self):
        mesh = self.mesh
        
        self.gradw1[:,:] = 0.
        self.gradw2[:,:] = 0.
        
        self.res[:,:] = 0.
        self.wsn[:] = 0.0
        
        self.gradw[:,:,:] = 0.0
        
        #----------------------------------------------------------------------
        # Compute gradients at cells.
        if (self.second_order): self.compute_gradients()
        if (self.use_limiter): self.compute_limiter()
        #----------------------------------------------------------------------
        
        
        #----------------------------------------------------------------------
        # Residual computation: interior faces
        #----------------------------------------------------------------------
        # Flux computation across internal faces (to be accumulated in res(:))
        #
        #          v2=Left(2)
        #        o---o---------o       face(j,:) = [i,k,v2,v1]
        #       .    .          .
        #      .     .           .
        #     .      .normal      .
        #    .  Left .--->  Right  .
        #   .   c1   .       c2     .
        #  .         .               .
        # o----------o----------------o
        #          v1=Right(1)
        #
        #
        # 1. Extrapolate the solutions to the face-midpoint from centroids 1 and 2.
        # 2. Compute the numerical flux.
        # 3. Add it to the residual for 1, and subtract it from the residual for 2.
        #
        #----------------------------------------------------------------------
        for face in mesh.faceList:
            adj_face = face.adjacentface
            
            c1 = face.parentcell     # Left cell of the face
            c2 = adj_face.parentcell # Right cell of the face
            
            v1 = face.nodes[0] # Left node of the face
            v2 = face.nodes[1] # Right node of the face
            
            u1 = self.u[c1.cid] #Conservative variables at c1
            u2 = self.u[c2.cid] #Conservative variables at c2
            
            self.gradw1 = self.gradw[c1.cid]
            self.gradw2 = self.gradw[c2.cid]
            
            unit_face_normal = face.normal_vector
            #Face midpoint at which we compute the flux.
            xm,ym = face.center
            
            
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
        return w
    
    
    
    #**************************************************************************
    # Compute limiter functions
    #
    #**************************************************************************
    def compute_limiter(self):
        # loop cells
        for cell in self.mesh.cellList:
            # loop primitive variables
            for i in range(nq):
                #----------------------------------------------------
                # find the min and max values
                # Initialize them with the solution at the current cell.
                # which could be min or max.
                wmin = self.w[cell.cid,i]
                wmax = self.w[cell.cid,i]
                
                #Loop over LSQ neighbors and find min and max
                for k in 
                
        return
    
    # survey of gradient reconstruction methods
    # https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20140011550.pdf
    def compute_gradients(self):
        return