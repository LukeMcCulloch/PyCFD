#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:15:20 2020

@author: lukemcculloch
"""
import numpy as np
import matplotlib.pyplot as plt
pi = np.pi

from flux import roe
from System2D import Grid

nq = 4 # Euler system size

class StencilLSQ(object):
    """
    #------------------------------------------
    #>> Cell-centered LSQ stencil data
    #------------------------------------------
    """
    def __init__(self, cell, mesh):
        self.cell = cell #reference to cell
        self.mesh = mesh #reference to mesh
        #
        self.nnghbrs_lsq = None     #number of lsq neighbors
        self.nghbr_lsq = []         #list of lsq neighbors
        self.cx = None              #LSQ coefficient for x-derivative
        self.cy = None              #LSQ coefficient for y-derivative
        #
        #self.node   = np.zeros((self.nNodes),float) #node to cell list
        self.construct_vertex_stencil()
        
        
    def construct_vertex_stencil(self):
        for node in self.cell.nodes:
            for cell in node.parent_cells:
                if cell is not self.cell:
                    self.nghbr_lsq.append(cell)
        
        self.nghbr_lsq = set(self.nghbr_lsq)
        self.nghbr_lsq = list(self.nghbr_lsq)
        self.nnghbrs_lsq = len(self.nghbr_lsq)
        return
    
    
    def plot_lsq_reconstruction(self, canvas = None,
                   alpha=.1):
        if canvas is None:
            fig, ax = plt.subplots()
            ax.axis('equal')
        else:
            ax = canvas
            
        ax = self.cell.plot_cell(canvas = ax,
                                 fillcolor=True)
        for cell in self.nghbr_lsq:
            ax = cell.plot_cell(canvas = ax)
        
        return
    

class Solvers(object):
    def __init__(self, mesh):
        self.mesh = mesh
        self.dim = mesh.dim
        
        self.second_order = True
        self.use_limiter = True
        
        # solution data
        self.u = np.zeros((mesh.nCells,nq),float) # conservative variables at cells/nodes
        self.w = np.zeros((mesh.nCells,nq),float) # primative variables at cells/nodes
        self.gradw = np.zeros((mesh.nCells,nq,self.dim),float) # gradients of w at cells/nodes.
        # 
        # solution convergence
        self.res = np.zeros((mesh.nCells,nq),float) #residual vector
        self.res_norm = np.zeros((nq,1),float)
        #
        # local convergence storage saved for speed
        self.gradw1 = np.zeros((nq,self.dim),float)
        self.gradw2 = np.zeros((nq,self.dim),float)
        
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
        
        
        #------------------------------------------
        #>> least squared gradient
        #------------------------------------------
        self.cclsq  = np.asarray( [StencilLSQ(cell,mesh) for cell in mesh.cells] )
        #e.g.
        #self.cclsq[0].nghbr_lsq #bulk list of all cells in the 'extended cell halo'
    
        #------------------------------------------
        #>> precompute least squared gradient coefficients
        #------------------------------------------
        self.compute_lsq_coefficients()
        self.test_lsq_coefficients()
        
        
    def solver_boot(self):
        
        #self.compute_lsq_coefficients()
        
        #self.set_initial_solution
        
        self.explicit_steady_solver()
        return
        
    
    
    def compute_lsq_coefficients(self):
        
        print "--------------------------------------------------"
        print " Computing LSQ coefficients... "
        
        ix = 0
        iy = 1
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        #The power to the inverse distance weight. The value 0.0 is used to avoid
        #instability known for Euler solvers. So, this is the unweighted LSQ gradient.
        #More accurate gradients are obtained with 1.0, and such can be used for the
        #viscous terms and source terms in turbulence models.
        lsq_weight_invdis_power = 1.0
        
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        # compute the LSQ coefficients (cx, cy) in all cells
        for i in range(self.mesh.nCells):
            cell = self.mesh.cells[i]
            #------------------------------------------------------------------
            #Define the LSQ problem size
            m = self.cclsq[i].nnghbrs_lsq
            n = self.dim
            
            
            #------------------------------------------------------------------
            # Allocate LSQ matrix and the pseudo inverse, R^{-1}*Q^T.
            a = np.zeros((m,n),float)
            #rinvqt  = np.zeros((n,m),float)
            
            #------------------------------------------------------------------
            # Build the weighted-LSQ matrix A(m,n).
            #
            #     weight_1 * [ (x1-xi)*wxi + (y1-yi)*wyi ] = weight_1 * [ w1 - wi ]
            #     weight_2 * [ (x2-xi)*wxi + (y2-yi)*wyi ] = weight_2 * [ w2 - wi ]
            #                 .
            #                 .
            #     weight_m * [ (xm-xi)*wxi + (ym-yi)*wyi ] = weight_2 * [ wm - wi ]
            for k, nghbr_cell in enumerate(self.cclsq[i].nghbr_lsq):
                dX = nghbr_cell.centroid - cell.centroid 
                # note you already stored this when you implemented this
                # in the mesh itself.
                
                weight_k = 1.0/(np.linalg.norm(dX)**lsq_weight_invdis_power)
                
                a[k,0] = weight_k*dX[0]
                a[k,1] = weight_k*dX[1]
                
            #------------------------------------------------------------------
            # Perform QR factorization and compute R^{-1}*Q^T from A(m,n)
            q, r = np.linalg.qr(a)
            rinvqt = np.dot( np.linalg.inv(r), q.T)
                
            #------------------------------------------------------------------
            #  Compute and store the LSQ coefficients: R^{-1}*Q^T*w
            #
            # (wx,wy) = R^{-1}*Q^T*RHS
            #         = sum_k (cx,cy)*(wk-wi).
            for k, nghbr_cell in enumerate(self.cclsq[i].nghbr_lsq):
                dX = nghbr_cell.centroid - cell.centroid 
                weight_k = 1.0/(np.linalg.norm(dX)**lsq_weight_invdis_power)
                self.cclsq[i].cx = rinvqt[ix,k] * weight_k
                self.cclsq[i].cy = rinvqt[iy,k] * weight_k
        return
    
    def test_lsq_coefficients(self, tol=1.e-10):
        """
          Compute the gradient of w=2*x+y 
          to see if we get wx=2 and wy=1 correctly.
        """
        verifcation_error = False
        
        for i, cell in enumerate(self.mesh.cells):
            
            #initialize wx and wy
            wx,wy = 0.0,0.0
            
            # (xi,yi) to be used to compute the function 2*x+y at i.
            xi,yi = cell.centroid
            
            #Loop over the vertex neighbors.
            for k, nghbr_cell in enumerate(self.cclsq[i].nghbr_lsq):
                
                #(xk,yk) to be used to compute the function 2*x+y at k.
                xk,yk = nghbr_cell.centroid
                
                # This is how we use the LSQ coefficients: 
                # accumulate cx*(wk-wi) and cy*(wk-wi).
                wx += self.cclsq[i].cx * ( (2.0*xk+yk) - (2.0*xi+yi))
                wy += self.cclsq[i].cy * ( (2.0*xk+yk) - (2.0*xi+yi))
            
            if (abs(wx-2.0) > tol) or (abs(wy-1.0) > tol) :
                print " wx = ", wx, " exact ux = 2.0"
                print " wy = ", wy, " exact uy = 1.0"
                verifcation_error = True
                
        if verifcation_error:
            print " LSQ coefficients are not correct. See above. Stop."
        else:
            print " Verified: LSQ coefficients are exact for a linear function."
        return
    
    
        
    #-------------------------------------------------------------------------#
    # Euler solver: Explicit Unsteady Solver: Ut + Fx + Gy = S
    #
    # This subroutine solves an un steady problem by 2nd-order TVD-RK with a
    # global time step.
    #-------------------------------------------------------------------------#
    def explicit_unsteady_solver(self, tfinal=None):
        time = 0.0
        if tfinal is None:
            self.t_final = 1.0
        else:
            self.t_final = tfinal
        
        while (time < self.t_final):
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
            
            #Set limiter functions
            if (self.use_limiter) :
                phi1 = self.phi[c1.cid]
                phi2 = self.phi[c2.cid]
            else:
                phi1 = 1.0
                phi2 = 1.0
                
            # Reconstruct the solution to the face midpoint and compute a numerical flux.
            # (reconstruction is implemented inside "interface_flux".
            self.interface_flux()
            
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
        
        w[self.ir] = u[0]
        w[self.iu] = u[1]/u[0]
        w[self.iv] = u[2]/u[0]
        w[self.ip] = (self.gamma-1.0)*( u[3] - \
                                       0.5*w[0]*(w[1]*w[1] + w[2]*w[2]) )
        return w
    
    
    
    #**************************************************************************
    # Compute limiter functions
    #
    #**************************************************************************
    def compute_limiter(self):
        # loop cells
        for cell in self.mesh.cells:
            i = cell.cid
            
            # loop primitive variables
            for ivar in range(nq):
                
                #----------------------------------------------------
                # find the min and max values
                # Initialize them with the solution at the current cell.
                # which could be min or max.
                wmin = self.w[cell.cid,ivar]
                wmax = self.w[cell.cid,ivar]
                
                #Loop over LSQ neighbors and find min and max
                for nghbr_cell_cid in self.cclsq[i].nghbr_lsq:
                    wmin = min(wmin, self.w[nghbr_cell_cid,ivar])
                    wmax = max(wmax, self.w[nghbr_cell_cid,ivar])
                
                #----------------------------------------------------
                # Compute phi to enforce maximum principle at vertices (MLP)
                xc,yc = self.mesh.cells[i].centroid
                
                # Loop over vertices of the cell i: 3 or 4 vertices for tria or quad.
                for k,iv in enumerate(self.mesh.cells[i].nodes):
                    xp,yp = iv.vector
                    
                    # Linear reconstruction to the vertex k
                    #diffx = xp-xc
                    #diffy = yp-yc
                    wf = self.w[i,ivar] + \
                                    self.gradw[i,ivar,0]*(xp-xc) + \
                                    self.gradw[i,ivar,1]*(yp-yc)
                    
                    # compute dw^-.
                    dwm = wf - self.w[i,ivar]
                    
                    # compute dw^+.
                    if ( dwm > 0.0 ):
                        dwp = wmax - self.w[i,ivar]
                    else:
                        dwp = wmin - self.w[i,ivar]
                    
                    # Increase magnitude by 'limiter_beps' without changin sign.
                    # dwm = sign(one,dwm)*(abs(dwm) + limiter_beps)
                    
                    # Note: We always have dwm*dwp >= 0 by the above choice! So, r=a/b>0 always
                    
                    # Limiter function: Venkat limiter
                    phi_vertex = self.vk_limiter(dwp, dwm, self.mesh.cells[i].volume)
                    
                    # Keep the minimum over the control points (vertices)
                    if (k==0):
                        phi_vertex_min = phi_vertex
                    else:
                        phi_vertex_min = min(phi_vertex_min, phi_vertex)
                        
                    #end of vertex loop
                    
                    
                # Keep the minimum over variables.
                if (ivar==0) :
                    phi_var_min = phi_vertex_min
                else:
                    phi_var_min = min(phi_var_min, phi_vertex_min)
                
                #end primative variable loop
            
            #Set the minimum phi over the control points and over the variables to be
            #our limiter function. We'll use it for all variables to be on a safe side.
            self.phi[i] = phi_var_min
            # end cell loop
        
        return
    
    def vk_limiter(self, a, b, vol):
        """
        ***********************************************************************
        * -- Venkat Limiter Function--
        *
        * 'Convergence to Steady State Solutions of the Euler Equations on Unstructured
        *  Grids with Limiters', V. Venkatakrishnan, JCP 118, 120-130, 1995.
        *
        * The limiter has been implemented in such a way that the difference, b, is
        * limited in the form: b -> vk_limiter * b.
        *
        * ---------------------------------------------------------------------
        *  Input:     a, b     : two differences
        *
        * Output:   vk_limiter : to be used as b -> vk_limiter * b.
        * ---------------------------------------------------------------------
        *
        ***********************************************************************
        """
        two = 2.0
        half = 0.5
        Kp = 5.0   #<<<<< Adjustable parameter K
        diameter = two*(vol/pi)**half
        eps2 = (Kp*diameter)**3
        vk_limiter = ( (a**2 + eps2) + two*b*a ) /                       \
                        (a**2 + two*b**2 + a*b + eps2)
        return vk_limiter
    
    
    # survey of gradient reconstruction methods
    # https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20140011550.pdf
    def compute_gradients(self):
        """
        #*******************************************************************************
        # Compute the LSQ gradients in all cells for all primitive variables.
        #
        # - Compute the gradient by [wx,wy] = sum_nghbrs [cx,cy]*(w_nghbr - wj),
        #   where [cx,cy] are the LSQ coefficients.
        #
        #*******************************************************************************
        """        
        #init gradient to zero
        self.gradw[:,:,:] = 0.
        
        # compute gradients for primative variables
        for ivar in range(nq):
            
            #compute gradients in all cells
            for cell in self.mesh.cells:
                i = cell.cid
                
                wi = self.w[i, ivar]
                
                #loop nieghbors
                for k in self.cclsq[i].nnghbrs_lsq:
                    nghbr_cell = self.cclsq[i].nghbr_lsq[k]
                    wk = self.w[nghbr_cell,ivar]    #Solution at the neighbor cell.
                    
                    self.gradw[i,ivar,0] = self.gradw[i,ivar,0] + self.cclsq[i].cx[k]*(wk-wi)
                    self.gradw[i,ivar,1] = self.gradw[i,ivar,1] + self.cclsq[i].cy[k]*(wk-wi)
        return
    
    
    def interface_flux(self):
        return roe(nx,gamma,uL,uR,f,fL,fR)
    
    
    
def show_LSQ_grad_area_plots():
    for cc in ssolve.cclsq[55:60]:
        cc.plot_lsq_reconstruction()
    return

if __name__ == '__main__':
    gd = Grid(type_='rect',m=10,n=10,
              winding='cw')
    self = Grid(type_='tri',m=10,n=10,
              winding='cw')
    
    cell = self.cellList[44]
    face = cell.faces[0]
    
    #cell.plot_cell()
    
    ssolve = Solvers(mesh = self)
    
    cc = ssolve.cclsq[33]
    cc.plot_lsq_reconstruction()
    
    show_LSQ_grad_area_plots()
    