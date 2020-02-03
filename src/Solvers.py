#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:15:20 2020

@author: lukemcculloch
"""
import os
import weakref
try:
    from memory_profiler import profile
    MEM_PROFILE = True
except:
    print 'please install memory_profiler'
    MEM_PROFILE = False
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

pi = np.pi

from flux import roe
from System2D import Grid
from BoundaryConditions import BC_states
from Parameters import Parameters

from Utilities import default_input
from DataHandler import DataHandler


nq = 4 # Euler system size

class StencilLSQ(object):
    """
    #------------------------------------------
    #>> Cell-centered LSQ stencil data
    #------------------------------------------
    """
    def __init__(self, cell, mesh):
        self.cell = cell #reference to cell
        #self.mesh = mesh #reference to mesh
        self._mesh = weakref.ref(mesh) if mesh else mesh
        #
        self.nnghbrs_lsq = None     #number of lsq neighbors
        self.nghbr_lsq = []         #list of lsq neighbors
        self.cx = []                #LSQ coefficient for x-derivative
        self.cy = []                #LSQ coefficient for y-derivative
        #
        #self.node   = np.zeros((self.nNodes),float) #node to cell list
        self.construct_vertex_stencil()
        
    @property
    def mesh(self):
        if not self._mesh:
            return self._mesh
        _mesh = self._mesh()
        if _mesh:
            return _mesh
        else:
            raise LookupError("mesh was destroyed")
            
    def __del__(self):
        print("delete LSQ",self.cell.cid)
    #    #print("delete", "LSQstencil")
        
        
    def construct_vertex_stencil(self):
        for node in self.cell.nodes:
            for cell in node.parent_cells:
                if cell is not self.cell:
                    self.nghbr_lsq.append(cell)
        
        self.nghbr_lsq = set(self.nghbr_lsq)
        self.nghbr_lsq = list(self.nghbr_lsq)
        self.nnghbrs_lsq = len(self.nghbr_lsq)
        
        # Allocate the LSQ coeffient arrays for the cell i:
        self.cx = np.zeros((self.nnghbrs_lsq),float)
        self.cy = np.zeros((self.nnghbrs_lsq),float)
        return
    
    
    def plot_lsq_reconstruction(self, canvas = None,
                                alpha = .1, saveit = False):
        if canvas is None:
            fig, ax = plt.subplots()
            ax.axis('equal')
        else:
            ax = canvas
            
        fig.suptitle('LSQ reconstruction stencil', fontsize=10)
            
        ax = self.cell.plot_cell(canvas = ax,
                                 fillcolor='green')
        for cell in self.nghbr_lsq:
            ax = cell.plot_cell(canvas = ax)
            
        patch = mpatches.Patch(color='green', label='primary cell')
        plt.legend(handles=[patch])
        
        if saveit:
            mytitle = '../pics/stencil_'+str(self.cell.cid)
            
            self.save_image(filename=mytitle, ftype = '.png')
        return
    
    
    def save_image(self, filename = None, ftype = '.pdf', closeit=True):
        """ save pdf file.
        No file extension needed.
        """
        if filename == None:
            filename = default_input('please enter a name for the picture', 'lsq_reconstruction')
        plt.savefig(filename+ftype, bbox_inches = 'tight')
        if closeit:
            plt.close()
        return
    

class Solvers(object):
    """
      2D Euler/NS equations = 4 equations:
     
      (1)continuity
      (2)x-momentum
      (3)y-momentum
      (4)energy
    """
    def __init__(self, mesh):
        self.solver_initialized = False
        self.mesh = mesh
        self.dim = mesh.dim
        
        #--------------------------------------
        # for the moment, default to simple initial conditions
        self.bc_type = ["freestream" for el in range(mesh.nBoundaries)] #np.zeros(mesh.nBoundaries, str)
        self.BC = BC_states(solver = self, flowstate = FlowState() ) 
        
        self.Parameters = Parameters()
        
        
        self.second_order = True
        self.use_limiter = True
        
        # solution data
        self.u = np.zeros((mesh.nCells,nq),float) # conservative variables at cells/nodes
        self.w = np.zeros((mesh.nCells,nq),float) # primative variables at cells/nodes
        self.gradw = np.zeros((mesh.nCells,nq,self.dim),float) # gradients of w at cells/nodes.
        #
        self.u0 = np.zeros((mesh.nCells,nq),float) #work array
        #
        # solution convergence
        self.res = np.zeros((mesh.nCells,nq),float) #residual vector
        self.res_norm = np.zeros((nq,1),float)
        #
        # local convergence storage saved for speed
        self.gradw1 = np.zeros((nq,self.dim),float)
        self.gradw2 = np.zeros((nq,self.dim),float)
        
        # update (pseudo) time step data
        #self.u0 = np.zeros((mesh.nCells,nq),float)
        self.dtau = np.zeros((mesh.nCells),float)
        
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
        
        #flux
        self.uL3d = np.zeros(5,float)       #conservative variables in 3D
        self.uR3d = np.zeros(5,float)       #conservative variables in 3D
        self.n12_3d = np.zeros(3,float)     #face normal in 3D
        self.num_flux3d = np.zeros(5,float) #numerical flux in 3D
        self.wsn = np.zeros((self.mesh.nCells),float) # max wave speed array
        
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
        
        
        #------------------------------------------
        #>> residual data
        #------------------------------------------
        self.num_flux = np.zeros(4,float)
        self.ub = np.zeros(4,float)
        self.wave_speed = 0.
        
        # local copies of data
        self.unit_face_normal = np.zeros((2),float)
        
        #------------------------------------------
        #>> exact solution data
        #------------------------------------------
        self.w_initial = np.zeros(4, float)
        
        
        
    def solver_boot(self, flowtype = 'vortex'):
        
        #self.compute_lsq_coefficients()
        
        def NotImp():
            print("not implemented yet")
            return
        
        
        switchdict = {
            'vortex':   self.initial_condition_vortex,
            'freestream': NotImp #self.set_initial_solution()
            }
        #switchdict.get(flowtype, "not implemented, at all")
        switchdict[flowtype]()
        
        #self.explicit_steady_solver()
        #self.explicit_unsteady_solver()
        
        self.solver_initialized = True
        return
    
    def solver_solve(self, tfinal=1.0, dt=.01):
        if not self.solver_initialized :
            print("You must initialize the solver first!")
            print("call solver_boot() on this object to initialize solver")
            return
        #self.explicit_steady_solver()
        self.explicit_unsteady_solver(tfinal=tfinal, dt=dt)
        return
        
    
    
    def compute_lsq_coefficients(self):
        """
        compute the neighbor-stencil-coefficients such that
        a gradient summed around a cell 
        (compact or extended stencil around the cell in questions)
        will give a least squares reconstruction of the gradient 
        at the cell in question
        """
        
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
                self.cclsq[i].cx[k] = rinvqt[ix,k] * weight_k
                self.cclsq[i].cy[k] = rinvqt[iy,k] * weight_k
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
                wx += self.cclsq[i].cx[k] * ( (2.0*xk+yk) - (2.0*xi+yi))
                wy += self.cclsq[i].cy[k] * ( (2.0*xk+yk) - (2.0*xi+yi))
            
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
    def explicit_unsteady_solver(self, tfinal=1.0, dt=.01):
        """
        
        debugging:
           
        self.t_final = 1.0
        time = 0.0
            
        """
        time = 0.0
        
        self.t_final = tfinal
        
        #-----------------------------------------------------------------------------
        #-----------------------------------------------------------------------------
        # Physical time-stepping
        #-----------------------------------------------------------------------------
        #-----------------------------------------------------------------------------

        
        #for jj in range(1): #debugging!
        while (time < self.t_final):
            print time
            #------------------------------------------------------------------
            # Compute the residual: res(i,:)
            self.compute_residual()
            
            #------------------------------------------------------------------
            # Compute the global time step, dt. One dt for all cells.
            dt = self.compute_global_time_step()
            
            #adjust time step?
            #code here
            
            #------------------------------------------------------------------
            # Increment the physical time and exit if the final time is reached
            time += dt #TBD dt was undefined
            
            #-------------------------------------------------------------------
            # Update the solution by 2nd-order TVD-RK.: u^n is saved as u0(:,:)
            #  1. u^*     = u^n - (dt/vol)*Res(u^n)
            #  2. u^{n+1} = 1/2*(u^n + u^*) - 1/2*(dt/vol)*Res(u^*)
            
            
            #-----------------------------
            #- 1st Stage of Runge-Kutta:
            #u0 = u: is solution data -  conservative variables at the cell centers I think
            
            self.u0[:] = self.u[:]
            # slow test first
            for i in range(self.mesh.nCells):
                self.u[i,:] = self.u0[i,:] - \
                                (dt/self.mesh.cells[i].volume) * self.res[i,:] #This is R.K. intermediate u*.
                self.w[i,:] = self.u2w( self.u[i,:]  )
                
                
            #-----------------------------
            #- 2nd Stage of Runge-Kutta:
                
            self.compute_residual()
            
            for i in range(self.mesh.nCells):
                self.u[i,:] = 0.5*( self.u[i,:] + self.u0[i,:] )  - \
                                0.5*(dt/self.mesh.cells[i].volume) * self.res[i,:]
                self.w[i,:] = self.u2w( self.u[i,:]  )
            
            
            
        print(" End of Physical Time-Stepping")
        print("---------------------------------------")
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
        
        # Gradients of primitive variables
        self.gradw1[:,:] = 0.
        self.gradw2[:,:] = 0.
        
        self.res[:,:] = 0.
        self.wsn[:] = 0.0
        
        self.gradw[:,:,:] = 0.0
        
        #----------------------------------------------------------------------
        # Compute gradients at cells
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
        savei = 0
        print 'do interior residual'
        for i,face in enumerate(mesh.faceList):
            """
            debugging:
            i = self.save[0]
            face = self.save[1]
            """
            #for i,face in enumerate(mesh.faceList[:2]):
            #TODO: make sure boundary faces are not in the 
            # main face list
            if face.isBoundary:
                pass
            else:
                savei = i
                adj_face = face.adjacentface
                
                c1 = face.parentcell     # Left cell of the face
                c2 = adj_face.parentcell # Right cell of the face
                
                v1 = face.nodes[0] # Left node of the face
                v2 = face.nodes[1] # Right node of the face
                
                u1 = self.u[c1.cid] #Conservative variables at c1
                u2 = self.u[c2.cid] #Conservative variables at c2
                
                self.gradw1 = self.gradw[c1.cid]
                self.gradw2 = self.gradw[c2.cid]
                
                self.unit_face_normal[:] = face.normal_vector[:]
                
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
                #print 'i = ',i
                num_flux, wave_speed = self.interface_flux(u1, u2,                     #<- Left/right states
                                                           self.gradw1, self.gradw2,   #<- Left/right same gradients
                                                           self.unit_face_normal,      #<- unit face normal
                                                           c1.centroid,                #<- Left cell centroid
                                                           c2.centroid,                #<- right cell centroid
                                                           xm, ym,                     #<- face midpoint
                                                           phi1, phi1,                 #<- Limiter functions
                                                           )
                test = np.any(np.isnan(num_flux)) or np.isnan(wave_speed)
                if test:
                    self.save = [i, face]
                assert(not test), "Found a NAN in interior residual"
                """
                debugging:
                    
                print u1, u2
                print self.gradw1
                print self.gradw2
                print self.unit_face_normal
                print c1.centroid
                print c2.centroid
                print xm,ym
                print phi1,phi2
                """
                
                #print i, num_flux, wave_speed
                #  Add the flux multiplied by the magnitude of the directed area vector to c1.
    
                self.res[c1.cid,:] += num_flux * face.face_nrml_mag
                self.wsn[c1.cid] += wave_speed * face.face_nrml_mag
    
                #  Subtract the flux multiplied by the magnitude of the directed area vector from c2.
                #  NOTE: Subtract because the outward face normal is -n for the c2.
                
                self.res[c2.cid,:] -= num_flux * face.face_nrml_mag
                self.wsn[c2.cid] += wave_speed * face.face_nrml_mag
    
                # End of Residual computation: interior faces
                #--------------------------------------------------------------------------------
    
    
    
    
                #--------------------------------------------------------------------------------
                #--------------------------------------------------------------------------------
                #--------------------------------------------------------------------------------
                #--------------------------------------------------------------------------------
                #--------------------------------------------------------------------------------
                # Residual computation: boundary faces:
                #
                # Close the residual by looping over boundary faces and distribute a contribution
                # to the corresponding cell.
                
                # Boundary face j consists of nodes j and j+1.
                #
                #  Interior domain      /
                #                      /
                #              /\     o
                #             /  \   /
                #            / c1 \ /   Outside the domain
                # --o-------o------o
                #           j   |  j+1
                #               |   
                #               v Face normal for the face j.
                #
                # c = bcell, the cell having the boundary face j.
                #
        savei = 0
        print 'do boundary residual'
        for ib, bface in enumerate(self.mesh.boundaryList):
            """
            ib = 4
            bface = self.mesh.boundaryList[4]
            
            ib = 0
            bface = self.mesh.boundaryList[ib]
            
            ib+=1
            bface = self.mesh.boundaryList[ib]
            """
            savei = ib
            v1 = bface.nodes[0] # Left node of the face
            v2 = bface.nodes[1] # Right node of the face
            
            #Face midpoint at which we compute the flux.
            xm,ym = bface.center
            
            #Set limiter functions
            if (self.use_limiter) :
                phi1 = self.phi[c1.cid]
                phi2 = self.phi[c2.cid]
            else:
                phi1 = 1.0
                phi2 = 1.0
                
            #Cell having a boundary face defined by the set of nodes j and j+1.
            c1 = bface.parentcell
                
            u1 = self.u[c1.cid] #Conservative variables at c1
            self.gradw1 = self.gradw[c1.cid]
            
            self.unit_face_normal[:] = bface.normal_vector[:]
            
            
            #---------------------------------------------------
            # Get the right state (weak BC!)
            #print 'ib = ',ib
            self.BC.get_right_state(xm,ym, 
                                    u1, 
                                    self.unit_face_normal, 
                                    self.bc_type[ib], #CBD (could be done): store these on the faces instead of seperate
                                    self.ub)
            
            self.gradw2 = self.gradw2 #<- Gradient at the right state. Give the same gradient for now.
            
            
            #---------------------------------------------------
            # Compute a flux at the boundary face.
            
            num_flux, wave_speed = self.interface_flux(u1, u2,                      #<- Left/right states
                                                       self.gradw1, self.gradw2,    #<- Left/right same gradients
                                                       self.unit_face_normal,       #<- unit face normal
                                                       c1.centroid,                 #<- Left cell centroid
                                                       [xm, ym],                    #<- make up a right cell centroid
                                                       xm, ym,                      #<- face midpoint
                                                       phi1, phi1,                  #<- Limiter functions
                                                       )
            test = np.any(np.isnan(self.wsn))  or np.isnan(wave_speed)
            if test:
                self.save = [ib, bface]
            assert(not test), "Found a NAN in boundary residual"
            print ib, num_flux, wave_speed
            #Note: No gradients available outside the domain, and use the gradient at cell c
            #      for the right state. This does nothing to inviscid fluxes (see below) but
            #      is important for viscous fluxes.
            
            #Note: Set right centroid = (xm,ym) so that the reconstruction from the right cell
            #      that doesn't exist is automatically cancelled: wR=wb+gradw*(xm-xc2)=wb.


            #---------------------------------------------------
            #  Add the boundary contributions to the residual.
            self.res[c1.cid,:] += num_flux * face.face_nrml_mag
            self.wsn[c1.cid] += wave_speed * face.face_nrml_mag

            # no c2 on the boundary

            # End of Residual computation: exterior faces
            #------------------------------------------------------------------
            #
            #end  compute_residual
            #S*****************************************************************
        return
    
    
    
    #-------------------------------------------------------------------------#
    #
    # time stepping
    #
    #-------------------------------------------------------------------------#
    def compute_global_time_step(self):
        CFL = self.Parameters.CFL
        
        #Initialize dt with the local time step at cell 1.
        i = 1
        assert(abs(self.wsn[i]) > 0.),'wsn time step initilization div by zero'
        physical_time_step = CFL*self.mesh.cells[i].volume / ( 0.5*self.wsn[i] )
        
        
        return physical_time_step
    
    
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
        
        iu = self.iu
        iv = self.iv
        ir = self.ir
        ip = self.ip
        
        w[ir] = u[0]
        #w[iu] = u[1]/u[0]
        #w[iv] = u[2]/u[0]
        if (u[0] >0):
            w[iu] = u[1]/u[0]
            w[iv] = u[2]/u[0]
        w[ip] = (self.gamma-1.0)*( u[3] - \
                                       0.5*w[0]*(w[1]*w[1] + w[2]*w[2]) )
        return w
    
    #-------------------------------------------------------------------------#
    #
    # compute u from w
    # ------------------------------------------------------------------------#
    #  Input:  w =    primitive variables (rho,     u,     v,     p)
    # Output:  u = conservative variables (rho, rho*u, rho*v, rho*E)
    # ------------------------------------------------------------------------#
    #
    # Note:    E = p/(gamma-1)/rho + 0.5*(u^2+v^2)
    #
    #-------------------------------------------------------------------------#
    def w2u(self, w):
        
        u = np.zeros((nq),float)
        
        gamma = self.gamma
        
        iu = self.iu
        iv = self.iv
        ir = self.ir
        ip = self.ip
        
        u[0] = w[ir]
        u[1] = w[ir]*w[iu]
        u[2] = w[ir]*w[iv]
        u[3] = w[ip]/(gamma-1.0)+0.5*w[ir]*(w[iu]*w[iu]+w[iv]*w[iv])
        return u
    
    
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
                for nghbr_cell in self.cclsq[i].nghbr_lsq:
                    wmin = min(wmin, self.w[nghbr_cell.cid,ivar])
                    wmax = max(wmax, self.w[nghbr_cell.cid,ivar])
                
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
            for i, cell in enumerate(self.mesh.cells):
                ci = cell.cid
                
                wi = self.w[ci, ivar] #solution at this cell
                
                #loop nieghbors
                for k in range(self.cclsq[ci].nnghbrs_lsq):
                    nghbr_cell = self.cclsq[ci].nghbr_lsq[k]
                    wk = self.w[nghbr_cell.cid,ivar]    #Solution at the neighbor cell.
                    
                    self.gradw[ci,ivar,0] = self.gradw[ci,ivar,0] + self.cclsq[ci].cx[k]*(wk-wi)
                    self.gradw[ci,ivar,1] = self.gradw[ci,ivar,1] + self.cclsq[ci].cy[k]*(wk-wi)
        return
    
    
    def interface_flux(self,
                       u1, u2, 
                       gradw1, gradw2, 
                       n12,                 # Directed area vector (unit vector)
                       C1,            # left centroid
                       C2,            # right centroid
                       xm, ym,              # face midpoint
                       phi1, phi2,          # limiter
                       ):
        """
        outputs:
            num_flux,            # numerical flux (output)
            wsn                  # max wave speed at face 
            
            
            gradw1 = self.gradw1
            gradw2 = self.gradw2
            n12 = face.normal_vector
            C1 = c1.centroid
            C2 = c2.centroid
            
            
        """
        
        xc1, yc1 = C1
        xc2, yc2 = C2
        zero = 0.0
        inviscid_flux = roe
        
        # convert consertative to primitive variables
        # at centroids.
        w1 = self.u2w(u1) 
        w2 = self.u2w(u2)
        
        # Linear Reconstruction in the primitive variables
        # primitive variables reconstructed to the face wL, WR:
        
        #Cell 1 centroid to the face midpoint:
        wL = w1 + phi1 * (gradw1[:,0]*(xm-xc1) + gradw1[:,1]*(ym-yc1))
        
        #Cell 2 centroid to the face midpoint:
        wR = w2 + phi2 * ( gradw2[:,0]*(xm-xc2) + gradw2[:,1]*(ym-yc2) )
        
        # Store the reconstructed solutions as conservative variables.
        # Just becasue flux functions use conservative variables.
        uL = self.w2u(wL) #conservative variables computed from wL and wR.
        uR = self.w2u(wR) #conservative variables computed from wL and wR.

        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        # Define 3D solution arrays and a 3D face normal.
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------

        #Left state: 3D <- 2D

        self.uL3d[0] = uL[0]
        self.uL3d[1] = uL[1]
        self.uL3d[2] = uL[2]
        self.uL3d[3] = zero
        self.uL3d[4] = uL[3]

        #Right state: 3D <- 2D
        
        self.uR3d[0] = uR[0]
        self.uR3d[1] = uR[1]
        self.uR3d[2] = uR[2]
        self.uR3d[3] = zero
        self.uR3d[4] = uR[3]
        
        #Normal vector
        
        self.n12_3d[0] = n12[0]
        self.n12_3d[1] = n12[1]
        self.n12_3d[2] = zero

        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        #  Compute inviscid flux by 3D flux subroutines
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        #------------------------------------------------------------
        #  (1) Roe flux
        #------------------------------------------------------------
        #return inviscid_flux(nx,gamma,uL,uR,f,fL,fR)
        num_flux, wsn = inviscid_flux(self.uL3d,self.uR3d,self.n12_3d, 
                                 self.num_flux3d,self.wsn,self.gamma)
        return num_flux[:-1], wsn
        
    
    def initial_condition_vortex(self):
        """
        #*******************************************************************************
        # Set the initial solution for the inviscid vortex test case.
        #
        # We initialize the solution with the exact solution. 
        #
        # Note: The grid must be generated in the square domain defined by
        #
        #       [x,y] = [-20,10]x[-20,10]
        #
        #       Initially, the vortex is centered at (x,y)=(-10,-10), and will be
        #       convected to the origin at the final time t=5.0.
        #
        #*******************************************************************************        
        """
        print( "setting: initial_condition_vortex")
        GridLen = .5
        x0      = -0.5*GridLen
        y0      = -0.5*GridLen
        K       =  5.0
        alpha   =  1.0
        gamma   = self.gamma
        
        # Set free stream values (the input Mach number is not used in this test).
        self.rho_inf = 1.0
        self.u_inf = 2.0
        self.v_inf = 2.0
        self.p_inf = 1.0/gamma
        
        # Note: Speed of sound a_inf is sqrt(gamma*p_inf/rho_inf) = 1.0.
        for i, cell in enumerate(self.mesh.cells):
            
            x = cell.centroid[0] - x0
            y = cell.centroid[1] - y0
            r = np.sqrt(x**2 + y**2)
            
            self.w_initial[self.iu] =  self.u_inf - K/(2.0*pi)*y*np.exp(alpha*0.5*(1.-r**2))
            self.w_initial[self.iv] =  self.v_inf + K/(2.0*pi)*x*np.exp(alpha*0.5*(1.-r**2))
            temperature =    1.0 - K*(gamma-1.0)/(8.0*alpha*pi**2)*np.exp(alpha*(1.-r**2))
            self.w_initial[self.ir] = self.rho_inf*temperature**(  1.0/(gamma-1.0)) #Density
            self.w_initial[self.ip] = self.p_inf  *temperature**(gamma/(gamma-1.0)) #Pressure
            
            #Store the initial solution
            self.w[i,:] = self.w_initial[:]
            
            # Compute and store conservative variables
            self.u[i,:] = self.w2u( self.w[i,:] )
            
        return

class FlowState(object):
    
    def __init__(self, rho_inf=1., u_inf=1., v_inf=1., p_inf=1.):
        self.rho_inf = rho_inf
        self.u_inf =  u_inf
        self.v_inf = v_inf
        self.p_inf = p_inf
        return
    
def show_LSQ_grad_area_plots():
    for cc in ssolve.cclsq[55:60]:
        cc.plot_lsq_reconstruction()
    return

def show_one_tri_cell():
    cc = ssolve.cclsq[57]
    cc.plot_lsq_reconstruction()
    cell = cc.cell
    cell.plot_cell()
    return

def show_ont_quad_cell():
    ssolve = Solvers(mesh = gd)
    cc =  ssolve.cclsq[57]
    cc.plot_lsq_reconstruction()
    cell = cc.cell
    cell.plot_cell()
    return



class TestInviscidVortex(object):
    
    def __init__(self):
        # up a level
        uplevel = os.path.join(os.path.dirname( os.getcwd() ))
        path2vortex = uplevel+'\\cases\case_unsteady_vortex'
        self.DHandler = DataHandler(project_name = 'vortex',
                                       path_to_inputs_folder = path2vortex)
        
        
        pass
    

if __name__ == '__main__':
    gd = Grid(type_='rect',m=10,n=10,
              winding='ccw')
    mesh = Grid(type_='tri',m=10,n=10,
              winding='ccw')
    
    cell = mesh.cellList[44]
    face = cell.faces[0]
    
    #cell.plot_cell()
    
    self = Solvers(mesh = mesh)
    
    #cc = ssolve.cclsq[33]
    #cc.plot_lsq_reconstruction()
    
    
    #----------------------------
    # plot LSQ gradient stencils
    #show_LSQ_grad_area_plots()
    
    
    # cc = ssolve.cclsq[57]
    # cc.plot_lsq_reconstruction()
    # cell = cc.cell
    # cell.plot_cell()
    
    test_vortex = TestInviscidVortex()
    
    
    """
    self.solver_boot(flowtype = 'vortex')
    self.solver_solve( tfinal=.0048, dt=.01)
    """