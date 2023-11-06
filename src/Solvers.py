#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:15:20 2020

@author: lukemcculloch
"""
import os
import sys
import weakref
try:
    from memory_profiler import profile
    MEM_PROFILE = True
except:
    print( 'please install memory_profiler')
    MEM_PROFILE = False
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pylab import * #quiver...
#
import matplotlib.tri as tri #plot unstructured data
# see https://matplotlib.org/gallery/images_contours_and_fields/irregulardatagrid.html


pi = np.pi

from flux import roe3D, roe2D #NOTE!: roe3D operates on conservative variables u, roe2D operates on primative variables w
from System2D import Grid
from BoundaryConditions import BC_states
from ManufacturedSolutions import compute_manufactured_sol_and_f_euler
from Parameters import Parameters

from Utilities import default_input
from DataHandler import DataHandler


import FileTools as FT
from PlotGrids import PlotGrid

from Debugging import dbInterfaceFlux, dbRoeFlux


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
        pass
        #print("delete LSQ",self.cell.cid)
        #print("delete", "LSQstencil")
        
        
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
        self.do_mms = False
        self.mesh = mesh
        self.dim = mesh.dim
        self.dpn = self.dim + 2#rho, u,v,p
        
        #self.solver_switch = {'mms':mms_solver,
        #                      'explicit':self.explicit_unsteady_solver,
        #                      'implicit':None}
        
        self.iparams = self.mesh.dhandle.inputParameters #parse the dictionary of input.nml data that the mesher read in through dhandle.inputParameters
        self.Parameters = Parameters(self.iparams)
        self.aoa = self.Parameters.aoa
        self.M_inf = self.Parameters.M_inf
        self.CFL = self.Parameters.CFL
        self.use_limiter = self.Parameters.use_limiter
        self.second_order = self.Parameters.second_order
        self.eig_limiting_factor = self.Parameters.eig_limiting_factor
        self.inviscid_flux = self.Parameters.inviscid_flux
        self.compute_te_mms = self.Parameters.compute_te_mms
        
        
        #self.second_order = True
        #self.use_limiter = True
        
        # solution data
        self.u = np.zeros((mesh.nCells,nq),float) # conservative variables at cells/nodes
        self.w = np.zeros((mesh.nCells,nq),float) # primative variables at cells/nodes
        self.gradw = np.zeros((mesh.nCells,nq,self.dim),float) # gradients of w at cells/nodes.
        
        #source (forcing) data, if any
        self.f = np.zeros((mesh.nCells,nq),float)
        
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
        
        #--------------------------------------
        # initial conditions
        
        # self.bc_type = {}
        # global_counter = 0
        # for i, el in enumerate(mesh.boundcount):
        #     print('parsing boundary {}, number of components {}'.format(i, el))
        #     btype = mesh.bound[i].bc_type
        #     #local_counter = 0
        #     for j in range(bb.nbfaces):
        #         #mesh.nBoundaries[global_counter]
        #         self.bc_type[i*] = btype
        #         global_counter += 1
        # #print('b elements set = {} =?= mesh.nBoundaries {}'.format(global_counter, mesh.nBoundaries))
        
        
        self.bc_type = {}
        global_counter = 0
        for bound in mesh.bound:
            btype = bound.bc_type
            nfaces = bound.nbfaces
            for f in range(nfaces):
                self.bc_type[global_counter] = btype
                global_counter += 1
            
        
        #print('\n Boundary nodes:')
        #print('    segments = {}'.format(len(self.mesh.boundcount)) )
        #for i in range(len(self.mesh.boundcount)):
        #    print(' boundary, {},   bnodes = {}'.format(i,self.mesh.bound[i].bnode))
        #    print('                bfaces = {}'.format(self.mesh.bound[i].nbnodes-1))
            
        # count = 0
        # for ii, bb in enumerate(self.mesh.bound):
        #     for jj, node in enumerate(bb.bnode):
        #         #self.bc_type[node] = bb.bc_type
        #         self.bc_type[count] = bb.bc_type #yes, the index is 0,1,2,.. len(self.mesh.bound)
        #         count += 1
        
        
        # self.BC = BC_states(solver = self, 
        #                     flowstate = FlowState(self.rho_inf, 
        #                                           self.u_inf, 
        #                                           self.v_inf,
        #                                           self.p_inf) ) 
        
        
    def print_nml_data(self):
        # for data in [self.aoa,
        #              self.M_inf,
        #              self.CFL,
        #              self.use_limiter,
        #              self.second_order,
        #              self.eig_limiting_factor]:
        #     name = f'{data=}'.split('=')[0].split('.')[-1]
        #     print(name = data)
        
        print('----------------------------')
        print('input.nml data')
        print('aoa =',self.aoa)
        print('M_inf =',self.M_inf)
        print('CFL =',self.CFL)
        
        print('compute_te_mms = ',self.compute_te_mms)
        
              
        print('inviscid_flux =',self.inviscid_flux)
        print('eig_limiting_factor =',self.eig_limiting_factor)
        
        print('second_order =',self.second_order)
        print('use_limiter =',self.use_limiter)
        print('----------------------------')
        return
    
    def solver_boot(self, flowtype = 'vortex'):
        
        #self.compute_lsq_coefficients()
        
        def NotImp():
            print("not implemented yet")
            return
        
        
        switchdict = {
            'mms':self.initial_solution_freestream,
            'vortex':   self.initial_condition_vortex,
            'freestream': self.initial_solution_freestream,
            'airfoil':self.initial_solution_freestream,
            'cylinder':self.initial_solution_freestream,
            'shock-diffraction':self.initial_solution_shock_diffraction
            }
        #switchdict.get(flowtype, "not implemented, at all")
        if flowtype == 'mms': 
            pass
        else:
            switchdict[flowtype]()
        
        
        
        
        self.BC = BC_states(solver = self, 
                            flowstate = FlowState(self.rho_inf, 
                                                  self.u_inf, 
                                                  self.v_inf,
                                                  self.p_inf) ) 
        
        self.solver_initialized = True
        return
    
    def solver_boot_mms(self):
        
        self.do_mms = True
        self.solver_initialized = True
        return
    
    def solver_solve(self, tfinal=1.0, dt=.01, solver_type='explicit_unsteady_solver'):
        if not self.solver_initialized :
            print("You must initialize the solver first!")
            print("call solver_boot() on this object to initialize solver")
            return
        #self.explicit_steady_solver()
        #self.explicit_unsteady_solver(tfinal=tfinal, dt=dt)
        
        
        self.solver_switch = {'mms_solver':[],
                              'explicit_unsteady_solver':[tfinal, dt],
                              'explicit_steady_solver':[tfinal, dt],
                              'explicit_unsteady_solver_efficient_shockdiffraction':[tfinal,dt]}
        
        
        
        getattr(self, solver_type)(*self.solver_switch[solver_type])
        
        
        return
        
    
    
    def compute_lsq_coefficients(self):
        """
        compute the neighbor-stencil-coefficients such that
        a gradient summed around a cell 
        (compact or extended stencil around the cell in questions)
        will give a least squares reconstruction of the gradient 
        at the cell in question
        """
        
        print( "--------------------------------------------------" )
        print( " Computing LSQ coefficients... " )
        
        ix = 0
        iy = 1
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        #The power to the inverse distance weight. The value 0.0 is used to avoid
        #instability known for Euler solvers. So, this is the unweighted LSQ gradient.
        #More accurate gradients are obtained with 1.0, and such can be used for the
        #viscous terms and source terms in turbulence models.
        lsq_weight_invdis_power = 0.0
        #lsq_weight_invdis_power = 1.0
        
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        # compute the LSQ coefficients (cx, cy) in all cells
        #for i in range(self.mesh.nCells): #note: this original i loop also works
        for cell in self.mesh.cells:
            #cell = self.mesh.cells[i] #note: this also works with the original i loop
            i = cell.cid
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
                print( " wx = ", wx, " exact ux = 2.0" )
                print( " wy = ", wy, " exact uy = 1.0" )
                verifcation_error = True
                
        if verifcation_error:
            print(" LSQ coefficients are not correct. See above. Stop." )
        else:
            print(" Verified: LSQ coefficients are exact for a linear function." )
        return
    
    def mms_solver(self):
        print('call mms_solver')
        self.mms_truncation_error()
        return
    
    def mms_truncation_error(self):
        
        
        mesh = self.mesh
        nCells = mesh.nCells
        zero = 0.0
        
        heffv = mesh.heffv
        
        for i in range(self.mesh.nbound):
            print("  Dirichlet BC enforced: ", i, self.mesh.bound[i].bc_type )
        
        # zero a forcing term array:
        self.f[:,:] = zero
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        # loop over all cells
        #for i in range(self.mesh.nCells):
        #    cell = self.mesh.cells[i]
        for cell in mesh.cells:
            i = cell.cid
            # Compute and store w (exact soltion in primitive variables) and f:
            self.w[i,:], self.f[i,:] = compute_manufactured_sol_and_f_euler(cell.centroid[0],
                                                             cell.centroid[1],
                                                             self.f[i,:])
            
            
            print('i, w(c1) = ',i, self.w[i,:])
            print('i, f(c1) = ',i, self.f[i,:])
            
            # Compute conservative variables from primitive variables.
            
            #self.u[i,:] = self.w[i,:]
            self.u[i,:] = self.w2u(self.w[i,:])
        
        # Compute the residuals (=TE*volume), and store them in res(:,:).
        self.compute_residual()
        
        #Subtract the forcing term.
        #Note: Res is an integral of Fx+Gy. So, the forcing term is also integrated,
        #      and so multiplied by the cell volume.
        for cell in mesh.cells:
            i = cell.cid
            self.res[i,:] -= self.f[i,:]*cell.volume
        
        # Compute L1 norms and print them on screen.
        
        # Initialization
        norm1 = zero
        
        # Sum of absolute values:
        for cell in mesh.cells:
            i = cell.cid
            norm1 += abs(self.res[i,:]/cell.volume) #TE = Res/Volume.
            
        # Take an average; this is the TE.
        norm1 = norm1 / float(nCells)
        
        # Print the TE for all 4 equations.
        print(" -------------- Truncation error norm ------------------")
        print("conti {} \n x-mom  {} \n y-mom {} \n energy {} \n heffv {}".format(
            norm1[0],norm1[1],norm1[2],norm1[3],heffv) )
        
        return
        
    #-------------------------------------------------------------------------#
    # Euler solver: Explicit Unsteady Solver: Ut + Fx + Gy = S
    #
    # This subroutine solves an un steady problem by 2nd-order TVD-RK with a
    # global time step.
    #-------------------------------------------------------------------------#
    def explicit_unsteady_solver(self, tfinal=10.0, dt=.01, itermax=1000):
        """
        
        debugging:
           
        self.t_final = 1.0
        time = 0.0
            
        """
        print('call explicit_unsteady_solver')
        time = 0.0
        itercount = 0
        
        self.t_final = tfinal
        
        #-----------------------------------------------------------------------------
        #-----------------------------------------------------------------------------
        # Physical time-stepping
        #-----------------------------------------------------------------------------
        #-----------------------------------------------------------------------------
        
        #--------------------------------------------------------------------------------
        # First, make sure that normal mass flux is zero at all solid boundary nodes.
        # NOTE: Necessary because initial solution may generate the normal component.
        #--------------------------------------------------------------------------------
        #self.eliminate_normal_mass_flux()
        
        #for jj in range(1): #debugging!
        while (time < self.t_final and itercount < itermax):
            print(time)
            #------------------------------------------------------------------
            # Compute the residual: res(i,:)
            #print("stage 1 compute residual")
            self.compute_residual(roe3D)
            
            #sys.exit()
            
            self.compute_residual_norm()
            #print('res_norm = {}'.format(self.res_norm))
            
            #------------------------------------------------------------------
            # Compute the global time step, dt. One dt for all cells.
            dt = self.compute_global_time_step()#*.5
            
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
            #print("stage 2 compute residual")
            self.compute_residual(roe3D)
            #exit()
            for i in range(self.mesh.nCells):
                self.u[i,:] = 0.5*( self.u[i,:] + self.u0[i,:] )  - \
                                0.5*(dt/self.mesh.cells[i].volume) * self.res[i,:]
                self.w[i,:] = self.u2w( self.u[i,:]  )
            
            self.compute_residual_norm()
            print('res_norm = {}'.format(self.res_norm))
            
            itercount += 1
        print(" End of Physical Time-Stepping")
        print("---------------------------------------")
        return
    
    
    
    #-------------------------------------------------------------------------#
    # Euler solver: Explicit Unsteady Solver: Ut + Fx + Gy = S
    # (efficient shock-diffraction grid version)
    #
    # This subroutine solves an un steady problem by 2nd-order TVD-RK with a
    # global time step.
    #-------------------------------------------------------------------------#
    def explicit_unsteady_solver_efficient_shockdiffraction(self, tfinal=1.0, dt=.01):
        """
        
        debugging:
           
        self.t_final = 1.0
        time = 0.0
            
        """
        print('call explicit_unsteady_solver')
        time = 0.0
        
        self.t_final = tfinal
        
        
        #-----------------------------------------------------------------------------
        #-----------------------------------------------------------------------------
        # Physical time-stepping
        #-----------------------------------------------------------------------------
        #-----------------------------------------------------------------------------
        
        #--------------------------------------------------------------------------------
        # First, make sure that normal mass flux is zero at all solid boundary nodes.
        # NOTE: Necessary because initial solution may generate the normal component.
        #--------------------------------------------------------------------------------
        self.eliminate_normal_mass_flux()
        
        #for jj in range(1): #debugging!
        i_iteration = 0
        while (time < self.t_final):
            print(time)
            #------------------------------------------------------------------
            # Compute the residual: res(i,:)
            #print("stage 1 compute residual")
            self.compute_residual(roe2D)
            
            #sys.exit()
            
            self.compute_residual_norm()
            #print('res_norm = {}'.format(self.res_norm))
            
            
            #--- Initial (no solution update yet) ------------
            if (i_iteration == 0) :
                
                
                print('t, step, Density    X-momentum  Y-momentum   Energy')
                #print(" Iteration   max(res)    max(res)/max(res)_initial ")
                print(time, i_iteration, np.max( self.res_norm[:] ))
                
            #--- After the first solution upate ------------
            elif(i_iteration%1 == 0):
                print(time, i_iteration, np.max( self.res_norm[:] ) )
            
            #------------------------------------------------------------------
            # Compute the global time step, dt. One dt for all cells.
            dt = self.compute_global_time_step()#*.5
            
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
            #u0 = u: is solution data -save it-  conservative variables at the cell centers I think
            self.u0[:] = self.u[:]
            # slow test first
            for i in range(self.mesh.nCells):
                self.u[i,:] = self.u0[i,:] - \
                                (dt/self.mesh.cells[i].volume) * self.res[i,:] #This is R.K. intermediate u*.
                self.w[i,:] = self.u2w( self.u[i,:]  )
                
                
            #-----------------------------
            #- 2nd Stage of Runge-Kutta:
            #print("stage 2 compute residual")
            self.compute_residual(roe2D)
            #exit()
            for i in range(self.mesh.nCells):
                self.u[i,:] = 0.5*( self.u[i,:] + self.u0[i,:] )  - \
                                0.5*(dt/self.mesh.cells[i].volume) * self.res[i,:]
                self.w[i,:] = self.u2w( self.u[i,:]  )
            
            self.compute_residual_norm()
            i_iteration += 1
            
        print(" End of Physical Time-Stepping")
        print("---------------------------------------")
        return
    
    
    
        
    #-------------------------------------------------------------------------#
    # Euler solver: Explicit Steady Solver: Ut + Fx + Gy = S
    #
    # Explicit Steady Solver: Ut + Fx + Gy = S, Ut -> 0.
    #
    # This subroutine solves a steady problem by explicit schemes with the time
    # taken as a pseudo time, and thus we can use local time steps for speed up
    # the convergence (and a local-preconditioning method if implemented).
    #
    # In other words, it solves a global system of nonlinear residual equations:
    #   Res(U) = 0
    # by explicit iteration schemes.
    #
    #-------------------------------------------------------------------------#
    def explicit_steady_solver(self, 
                               tfinal=1.0, 
                               dt=.01, 
                               tol=1.e-5, 
                               max_iteration = 500):
        """
        
        debugging:
           
        self.t_final = 1.0
        time = 0.0
            
        """
        print('call explicit_steady_solver')
        time = 0.0
        
        self.t_final = tfinal
        self.max_iteration = max_iteration
        
        #-----------------------------------------------------------------------------
        #-----------------------------------------------------------------------------
        # Pseudo time-stepping
        #-----------------------------------------------------------------------------
        #-----------------------------------------------------------------------------
        print()
        print("---------------------------------------")
        print(" Pseudo time-Stepping")
        print('max_iteration = ',max_iteration)
        print()
        
        
        i_iteration = 0
        pseudo_time_loop = True
        
        while (pseudo_time_loop):
            #print(time)
            #------------------------------------------------------------------
            # Compute the residual: res(i,:) (gradient computation is done within)
            #print("stage 1 compute residual")
            self.compute_residual(roe3D)
            #sys.exit()
            
            #Compute the residual norm for checking convergence.
            self.compute_residual_norm()
            
            
            #print('res_norm = {}'.format(self.res_norm))
            
            
            #--- Initial (no solution update yet) ------------
            if (i_iteration == 0) :
                #Save the initial max res norm.
                res_norm_initial = np.copy(self.res_norm)
                
                print(" Iteration   max(res)    max(res)/max(res)_initial ")
                print(i_iteration, np.max( self.res_norm[:] ), 1.0)
                
            #--- After the first solution upate ------------
            else:
                print( i_iteration, np.max( self.res_norm[:] ),
                      np.max( self.res_norm[:] / res_norm_initial[:] )
                      )
                
                #print(i_iteration, np.max( self.res_norm[:]/res_norm_initial[:] ))
            
            
            #------------------------------------------------------
            # Exit if the res norm is reduced below the tolerance specified by input.
            
            if ( np.max( self.res_norm[:]/res_norm_initial[:] ) < tol ): pseudo_time_loop=False
    
            #------------------------------------------------------
            # Exit if we reach a specified number of iterations.
            if (i_iteration == max_iteration): pseudo_time_loop=False
            
    
            #------------------------------------------------------
            # Increment the counter and go to the next iteration.
            
            i_iteration = i_iteration + 1 #<- This is the iteration that has just been done.
            
            
            #------------------------------------------------------
            # Compute the local time step, dtau.
            self.compute_local_time_step_dtau()
            
            
            #------------------------------------------------------------------
            # Increment the physical time and exit if the final time is reached
            time += dt #TBD dt was undefined
            
            #------------------------------------------------------
            # Update the solution by forward Euler scheme.
            
            #-------------------------------------------------------------------
            # Update the solution by 2nd-order TVD-RK.: u^n is saved as u0(:,:)
            #  1. u^*     = u^n - (dt/vol)*Res(u^n)
            #  2. u^{n+1} = 1/2*(u^n + u^*) - 1/2*(dt/vol)*Res(u^*)
            
            
            #-----------------------------
            #- 1st Stage of Runge-Kutta:
            
            self.u0[:] = self.u[:]
            # slow test first
            for i in range(self.mesh.nCells):
                self.u[i,:] = self.u0[i,:] - \
                                (self.dtau[i]/self.mesh.cells[i].volume) * self.res[i,:] #This is R.K. intermediate u*.
                self.w[i,:] = self.u2w( self.u[i,:]  )
                
                
            #-----------------------------
            #- 2nd Stage of Runge-Kutta:
            #print("stage 2 compute residual")
            self.compute_residual(roe3D)
            #exit()
            for i in range(self.mesh.nCells):
                self.u[i,:] = 0.5*( self.u[i,:] + self.u0[i,:] )  - \
                                0.5*(self.dtau[i]/self.mesh.cells[i].volume) * self.res[i,:]
                self.w[i,:] = self.u2w( self.u[i,:]  )
                
            #pseudo_time_loop = False
            
        print(" End of Pseudo Time-Stepping")
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
        #self.res_norm[:] = np.sum(np.abs(self.res)) / float(self.mesh.nCells)
        for i in range(len(self.res[0,:])):
            self.res_norm[i] = np.linalg.norm(self.res[:,i]) / float(self.mesh.nCells)
        return
    
    #-------------------------------------------------------------------------#
    #
    # compute_residual: comptutes the residuals at cells for
    # the cell-centered finite-volume discretization.
    #
    #-------------------------------------------------------------------------#
    def compute_residual(self, flux=None):
        mesh = self.mesh
        
        if flux==None:
            flux = roe3D
            
        # Gradients of primitive variables
        self.gradw1[:,:] = 0.0
        self.gradw2[:,:] = 0.0
        
        self.res[:,:] = 0.0
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
        #savei = 0
        #print('do interior residual')
        #print('nfaces = ',len(self.mesh.faceList))
        for face in mesh.faceList:
            """
            #debugging:
            i = self.save[0]
            face = self.save[1]
            
            """
            #for i,face in enumerate(mesh.faceList[:2]):
            #TODO: make sure boundary faces are not in the 
            # main face list
            if face.isBoundary:
                #print('POSSIBLE ISSUE: boundary faces found on the interior iteration')
                pass
            else:
                #savei = i
                adj_face = face.adjacentface
                
                c1 = face.parentcell     # Left cell of the face
                c2 = adj_face.parentcell # Right cell of the face
                
                #v1 = face.nodes[0] # Left node of the face
                #v2 = face.nodes[1] # Right node of the face
                
                
                #print('v1 = ',v1.nid)
                #print('v2 = ',v2.nid)
                
                u1 = self.u[c1.cid] #Conservative variables at c1
                u2 = self.u[c2.cid] #Conservative variables at c2
                
                self.gradw1 = self.gradw[c1.cid] # Gradient of primitive variables at c1
                self.gradw2 = self.gradw[c2.cid] # Gradient of primitive variables at c2
                
                self.unit_face_normal[:] = face.normal_vector[:] # Unit face normal vector: c1 -> c2.
                
                #print('self.gradw1 = ',self.gradw1)
                #print('self.gradw2 = ',self.gradw2)
                #print('unit_face_normal = ',self.unit_face_normal)
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
                #print('i = ',i)
                num_flux, wave_speed = self.interface_flux(u1[:], u2[:],                     #<- Left/right states
                                                           self.gradw1, self.gradw2,   #<- Left/right same gradients
                                                           face.normal_vector,         #<- unit face normal
                                                           c1.centroid,                #<- Left cell centroid
                                                           c2.centroid,                #<- right cell centroid
                                                           xm, ym,                     #<- face midpoint
                                                           phi1, phi2,                 #<- Limiter functions
                                                           flux)
                
                #print('id, num_flux, wave_speed = ',c1.cid, num_flux, wave_speed)
                test = np.any(np.isnan(num_flux)) or np.isnan(wave_speed)
                # self.dbugIF = dbInterfaceFlux(u1, u2,                     #<- Left/right states
                #                     self.gradw1, self.gradw2,   #<- Left/right same gradients
                #                     face,                       #<- unit face //normal
                #                     c1,                         #<- Left cell // centroid
                #                     c2,                         #<- right cell // centroid
                #                     xm, ym,                     #<- face midpoint
                #                     phi1, phi2,                 #<- Limiter functions)
                #                     )
                assert(not test), "Found a NAN in interior residual"
                
                #  Add the flux multiplied by the magnitude of the directed area vector to c1.
    
                self.res[c1.cid,:] += num_flux * face.face_nrml_mag
                self.wsn[c1.cid] += wave_speed * face.face_nrml_mag
    
                #  Subtract the flux multiplied by the magnitude of the directed area vector from c2.
                #  NOTE: Subtract because the outward face normal is -n for the c2.
                
                self.res[c2.cid,:] -= num_flux * face.face_nrml_mag
                self.wsn[c2.cid] += wave_speed * face.face_nrml_mag
                
                # print('i, res(c1) = ',c1.cid, self.res[c1.cid,:])
                # print('i, res(c2) = ',c2.cid, self.res[c2.cid,:])
                # print('i, wsn(c1) = ',c1.cid, self.wsn[c1.cid])
                # print('i, wsn(c2) = ',c2.cid, self.wsn[c2.cid])
                
                # print('--------------------------')
    
                # End of Residual computation: interior faces
                #--------------------------------------------------------------------------------
                
    
    
    
        #--------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------
        # Residual computation: boundary faces:
        #
        # Close the residual by looping over boundary faces 
        # and distribute a contribution to the corresponding cell.
        
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
        #savei = 0
        #print('do boundary residual')
        for ib, bface in enumerate(self.mesh.boundaryList):
            """
            ib = self.save[0]
            bface = self.save[1]
            """
            #Cell having a boundary face defined by the set of nodes j and j+1.
            c1 = bface.parentcell
            
            #savei = ib
            v1 = bface.nodes[0] # Left node of the face
            v2 = bface.nodes[1] # Right node of the face
            
            # print('v1 = ',v1.nid)
            # print('v2 = ',v2.nid)
            
            #Face midpoint at which we compute the flux.
            xm,ym = bface.center
            
            #Set limiter functions
            if (self.use_limiter) :
                phi1 = self.phi[c1.cid]
                phi2 = 1.0
            else:
                phi1 = 1.0
                phi2 = 1.0
                
                
            u1 = self.u[c1.cid] #Conservative variables at c1
            self.gradw1 = self.gradw[c1.cid]
            
            self.unit_face_normal[:] = bface.normal_vector[:]
            
            #---------------------------------------------------
            # Get the right state (weak BC!)
            self.ub = self.BC.get_right_state(xm,ym, 
                                    u1[:], 
                                    self.unit_face_normal, 
                                    self.bc_type[ib], #CBD (could be done): store these on the faces instead of seperate  (tlm what?...cells?) 
                                    f=self.f[c1.cid])
            
            self.gradw2 = self.gradw2 #<- Gradient at the right state. Give the same gradient for now.
            
            
            
            # print('u1 = ',u1)
            #---------------------------------------------------
            # Compute a flux at the boundary face.
            # print('bface normal = ', self.unit_face_normal)
            # print('ub = ',self.ub)
            num_flux, wave_speed = self.interface_flux(u1[:], self.ub,                      #<- Left/right states
                                                       self.gradw1, self.gradw2,    #<- Left/right same gradients
                                                       self.unit_face_normal,       #<- unit face normal
                                                       c1.centroid,                 #<- Left cell centroid
                                                       [xm, ym],                    #<- Set right centroid = (xm,ym)
                                                       xm, ym,                      #<- face midpoint
                                                       phi1, phi2,                  #<- Limiter functions
                                                       flux)
            test = np.any(np.isnan(self.wsn))  or np.isnan(wave_speed)
            assert(not test), "Found a NAN in boundary residual"
            #print(c1.cid, num_flux, wave_speed)
            #Note: No gradients available outside the domain, and use the gradient at cell c
            #      for the right state. This does nothing to inviscid fluxes (see below) but
            #      is important for viscous fluxes.
            
            #Note: Set right centroid = (xm,ym) so that the reconstruction from the right cell
            #      that doesn't exist is automatically cancelled: wR=wb+gradw*(xm-xc2)=wb.
            # so assert(wR == wb)
            

            #---------------------------------------------------
            #  Add the boundary contributions to the residual.
            self.res[c1.cid,:] += num_flux * bface.face_nrml_mag
            self.wsn[c1.cid] += wave_speed * bface.face_nrml_mag

            # # no c2 on the boundary
            # print('face_nrml_mag = ',bface.face_nrml_mag)
            
            
            # print(' res(c1) = ', self.res[c1.cid,:])
            # print(' wsn(c1) = ', self.wsn[c1.cid])
            # print('------------------')

            # End of Residual computation: exterior faces
            #------------------------------------------------------------------
            #
            #end  compute_residual
            #S*****************************************************************
        
        # for bface in self.mesh.boundaryList:
        #     print(bface.parentcell.cid,bface.face_nrml_mag)
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
        
        for i, cell in enumerate(self.mesh.cells):
            physical_time_step = min( physical_time_step,
                        CFL*self.mesh.cells[i].volume / ( 0.5*self.wsn[i] )
                                     )
        
        return physical_time_step
    
    def compute_local_time_step_dtau(self):
        CFL = self.Parameters.CFL
        for i, cell in enumerate(self.mesh.cells):
            self.dtau[i] =  CFL*self.mesh.cells[i].volume / ( 0.5*self.wsn[i] )
        return 
    
    
    #********************************************************************************
    #* Prepararion for Tangency condition (slip wall):
    #*
    #* Eliminate normal mass flux component at all solid-boundary nodes at the
    #* beginning. The normal component will never be changed in the solver: the
    #* residuals will be constrained to have zero normal component.
    #*
    #********************************************************************************
    def eliminate_normal_mass_flux(self):
        
        only_slip_wall = False
        savei = -1
        for i, bg in enumerate(self.mesh.bound):
            if bg.bc_type == 'slip_wall': 
                only_slip_wall = True #use this to only set the boundary node on the single corner node!  (very specilaized gridding, indeed)
                savei = i
                print(" Eliminating the normal momentum on slip wall boundary ", i)
                
                
                
        for ib, bface in enumerate(self.mesh.boundaryList):
            if self.bc_type[ib] == 'slip_wall' and only_slip_wall:
                ################################################################
                # THIS IS A SPECIAL TREATMENT FOR SHOCK DIFFRACTION PROBLEM.
                #
                # NOTE: This is a corner point between the inflow boundary and
                #       the lower-left wall. Enforce zero y-momentum, which is
                #       not ensured by the standard BCs.
                #       This special treatment is necessary because the domain
                #       is rectangular (the left boundary is a straight ine) and
                #       the midpoint node on the left boundary is actually a corner.
                #
                #       Our computational domain:
                #
                #                 ---------------
                #          Inflow |             |
                #                 |             |  o: Corner node
                #          .......o             |
                #            Wall |             |  This node is a corner.
                #                 |             |
                #                 ---------------
                #
                #       This is to simulate the actual domain shown below:
                #      
                #         -----------------------
                # Inflow  |                     |
                #         |                     |  o: Corner node
                #         --------o             |
                #            Wall |             |
                #                 |             |
                #                 ---------------
                #      In effect, we're simulating this flow by a simplified
                #      rectangular domain (easier to generate the grid).
                #      So, an appropriate slip BC at the corner node needs to be applied,
                #      which is "zero y-momentum", and that's all.
                #
                
                # if (i==2 and j==1):
                #             inode = bound(i)%bnode(j)
                #  node(inode)%u(3) = zero               # Make sure zero y-momentum.
                #  node(inode)%w    = u2w(node(inode)%u) #Update primitive variables
                #  cycle bnodes_slip_wall # That's all we neeed. Go to the next.
                
                ################################################################
                cid = bface.parentcell.cid
                if only_slip_wall:
                    self.u[cid,2] = 0.0                      # Make sure zero y-momentum.
                    self.w[cid,:] = self.u2w(self.u[cid,:])  # Update primitive variables
                    only_slip_wall = False
                ################################################################
                else:
                    nij = bface.normal_vector
                    normal_mass_flux = self.u[cid,1]*nij[0] + self.u[cid,2]*nij[1]
                    
                    #zero the normal mass flux:
                    self.u[cid,1] = self.u[cid,1] - normal_mass_flux * nij[0]
                    self.u[cid,2] = self.u[cid,2] - normal_mass_flux * nij[1]
                    
                    self.w[cid,:] = self.u2w(self.u[cid,:])
        
        print(" Finished eliminating the normal momentum on slip wall boundary ", savei)
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
        '''
        Compute primitive variables from conservative variables.

        Parameters
        ----------
        u : conservative variables (rho, rho*u, rho*v, rho*E)

        Returns
        -------
        w : primitive variables (rho,     u,     v,     p)

        '''
        #print('u',u)
        nq=4
        w = np.zeros((nq),float)
        
        ir = self.ir
        iu = self.iu
        iv = self.iv
        ip = self.ip
        
        
        #if u[0] == 0.0: 
        #    u[0] = 1.0e-15#1.e15
        #    #print('setting u density to 1e-15 to fix devide by zero in u2w')
        
        w[ir] = u[0]
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
        '''
        Compute conservative variables from primitive variables.

        Parameters
        ----------
        w : primitive variables (rho,     u,     v,     p)

        Returns
        -------
        u : conservative variables (rho, rho*u, rho*v, rho*E)

        '''
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
        #print('compute_limiter')
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
    #
    # compose the finite difference 1st order gradient
    # from each element of the stencil.
    #
    # There are many more cells in the neighborhood than are needed to 
    # compute a gradient, so write the overdetermined 
    # system Ax=b 
    # 
    # where 
    # 
    # A is the matrix of spatial differences between noe centers
    #   (in this case an Ncells x 2D matrix)
    #
    # B is the vector of primative variable differences, phi_i - phi_o
    #   between the values at surrounding nodes, and the node in question
    #
    # x is just the finite difference we seek:
    #      [ d phi_o / d x , d phi_o / d y ] = ( A.T A ).inv A.T B
    #
    # Q: Why are we doing this?  
    #
    #       A: to extrapolate solutions linearly from the cell centroids
    #           to the faces (face midpoints)
    #
    # Q: Why are we doing that?
    #
    #       A: This slope will allow us to reconstruct 
    #           the fluxes at the cell boundaries in second order 
    #           accurate fashion.  (we will use limiters to achieve monotonicity)
    #
    #           Then bob's your uncle, solve the Riemann problem
    # 
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
        #print('compute_gradients')
        #init gradient to zero
        self.gradw[:,:,:] = 0.
        
        # compute gradients for primative variables
        for ivar in range(nq):
            
            #compute gradients in all cells
            for cell in self.mesh.cells:
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
                       inviscid_flux):
        """
        outputs:
            num_flux,            # numerical flux (output)
            wsn                  # max wave speed at face 
            
        
        
        debug inputs:
            
            #interior
            gradw1 = self.gradw1
            gradw2 = self.gradw2
            n12 = face.normal_vector
            C1 = c1.centroid
            C2 = c2.centroid
            
            
            #boundary
            gradw1 = self.gradw1
            gradw2 = self.gradw2
            n12 = face.normal_vector
            C1 = c1.centroid
            C2 = [xm, ym]
            
        
        
            
        """
        
        xc1, yc1 = C1
        xc2, yc2 = C2
        zero = 0.0
        #if inviscid_flux is None:
        #    inviscid_flux = roe #roe_primative
        
        # convert consertative to primitive variables at centroids.
        #print('u1 ',u1)
        #print('u2 ',u2)
        w1 = self.u2w(u1) 
        w2 = self.u2w(u2)
        
        # Linear Reconstruction in the primitive variables
        # primitive variables reconstructed to the face wL, WR:
        
        #Cell 1 centroid to the face midpoint:
        wL = w1[:] + phi1 * (gradw1[:,0]*(xm-xc1) + gradw1[:,1]*(ym-yc1))
        
        #Cell 2 centroid to the face midpoint:
        wR = w2[:] + phi2 * ( gradw2[:,0]*(xm-xc2) + gradw2[:,1]*(ym-yc2) )
        
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
        self.num_flux3d, wsn = inviscid_flux(self.uL3d, # conservative u's left cell off the face
                                              self.uR3d, # conservative u's right cell off the face
                                              self.n12_3d, #normal vector
                                              self.num_flux3d, #numerical flux
                                              self.wsn, #wave speed
                                              self.gamma)
        
        self.num_flux[0] = self.num_flux3d[0] # rho flux'
        self.num_flux[1] = self.num_flux3d[1] # mmtm-x flux'
        self.num_flux[2] = self.num_flux3d[2] # mmtm-y flux'
        self.num_flux[3] = self.num_flux3d[4] # energy flux
        return self.num_flux[:], wsn
        
    
    def initial_solution_freestream(self, M_inf = 0.3, aoa = 0.0):
        """
        #*******************************************************************************
        # Set the initial solution.
        #
        # We initialize the solution with a free stream condition defined by the Mach
        # number and the angle of attack specified by the input parameters: M_inf and aoa.
        #
        #*******************************************************************************     
        """
        print( "setting: set_initial_solution freestream")
        
        aoa = self.aoa
        
        M_inf = self.M_inf
            
            
        
            
        
        # Set free stream values based on the input Mach number.
        self.rho_inf = 1.0
        self.u_inf = M_inf*np.cos(aoa *np.pi/180.0) #aoa converted from degree to radian
        self.v_inf = M_inf*np.sin(aoa *np.pi/180.0) #aoa converted from degree to radian
        self.p_inf = 1.0/self.gamma
        
        self.w_initial[self.ir] =  self.rho_inf #Density
        self.w_initial[self.iu] =  self.u_inf   #u_inf
        self.w_initial[self.iv] =  self.v_inf   #v_inf
        self.w_initial[self.ip] =  self.p_inf   #Pressure 
        
        # Note: Speed of sound a_inf is sqrt(gamma*p_inf/rho_inf) = 1.0.
        for i, cell in enumerate(self.mesh.cells):
            
            
            #Store the initial solution
            self.w[i,:] = self.w_initial[:]
            
            # Compute and store conservative variables
            self.u[i,:] = self.w2u( self.w[i,:] )
        
        print('Done with freestream setup')
        return
        
    
    def initial_condition_vortex(self, vortex_strength=15.):
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
        #GridLen = 1.0
        x0      = -10.0 #0.5*GridLen
        y0      =  5.0 #0.5*GridLen
        K       =  vortex_strength
        alpha   =  1.0
        gamma   = self.gamma
        frac = 2.
        
        # Set free stream values (the input Mach number is not used in this test).
        self.rho_inf = 1.0
        self.u_inf = 1.0
        self.v_inf = 0.0
        self.p_inf = 1.0/gamma
        
        # Note: Speed of sound a_inf is sqrt(gamma*p_inf/rho_inf) = 1.0.
        for i, cell in enumerate(self.mesh.cells):
            
            x = cell.centroid[0] - x0
            y = cell.centroid[1] - y0
            r = np.sqrt(x**2 + y**2)
            
            self.w_initial[self.iu] =  self.u_inf - K/(frac*pi)*y*np.exp(alpha*0.5*(1.-r**2.))
            self.w_initial[self.iv] =  self.v_inf + K/(frac*pi)*x*np.exp(alpha*0.5*(1.-r**2.))
            temperature =    1.0 - K*(gamma-1.0)/(8.0*alpha*pi**2.)*np.exp(alpha*(1.-r**2.))
            self.w_initial[self.ir] = self.rho_inf*temperature**(  1.0/(gamma-1.0)) #Density
            self.w_initial[self.ip] = self.p_inf  *temperature**(gamma/(gamma-1.0)) #Pressure
            
            #Store the initial solution
            self.w[i,:] = self.w_initial[:]
            
            # Compute and store conservative variables
            self.u[i,:] = self.w2u( self.w[i,:] )
            
        return
    
    def initial_solution_shock_diffraction(self):
        '''
        ********************************************************************************
        * Initial solution for the shock diffraction problem:
        *
        * NOTE: So, this is NOT a general purpose subroutine.
        *       For other problems, specify M_inf in the main program, and
        *       modify this subroutine to set up an appropriate initial solution.
        
          Shock Diffraction Problem:
        
                                     Wall
                             --------------------
         Post-shock (inflow) |                  |
         (rho,u,v,p)_inf     |->Shock (M_shock) |            o: Corner node
            M_inf            |                  |
                      .......o  Pre-shock       |Outflow
                        Wall |  (rho0,u0,v0,p0) |
                             |                  |
                             |                  |
                             --------------------
                                   Outflow
        
        ********************************************************************************
        '''
        self.gamma = 1.4
        gamma = self.gamma
        
        # Pre-shock state: uniform state; no disturbance has reahced yet.
        one = 1.0
        zero = 0.0
        two = 2.0
        rho0 = one
        u0 = zero
        v0 = zero
        p0 = one/gamma
        
        #vertex or cell based?
        #for i, vtx in enumerate(self.mesh.nodes) #example is vertex (node) centered
        for i, cell in enumerate(self.mesh.cells): #this code is set up to be cell centered
            
            # Incoming shock speed
            
            M_shock = 5.09
            u_shock = M_shock * np.sqrt(gamma*p0/rho0)
            
            # Post-shock state: These values will be used in the inflow boundary condition.
            rho_inf = rho0 * (gamma + one)*M_shock**2/( (gamma - one)*M_shock**2 + two )
            p_inf =   p0 * (   two*gamma*M_shock**2 - (gamma - one) )/(gamma + one)
            u_inf = (one - rho0/rho_inf)*u_shock
            self.M_inf = u_inf / np.sqrt(gamma*p_inf/rho_inf)
            self.v_inf = zero
            
            
            self.u_inf  = u_inf
            self.p_inf = p_inf
            
            # Set the initial solution: set the pre-shock state inside the domain.
            
            #node[i].w = np.asarray([ rho0, u0, v0, p0 ])
            #node[i].u = self.w2u( node(i).w )
            
            self.w[i,:] = np.asarray([ rho0, u0, v0, p0 ])
            self.u[i,:] = self.w2u(self.w[i,:])
            
        return
    
    def write_solution(self):
        self.write_flow_at_cell_centers()
        return
    
    def plot_solution(self, title='Not Inital'):
        self.plot_flow_at_cell_centers(title)
        return
    
    def write_flow_at_cell_centers(self):
        self.solution_dir = '../pics/solution'
        location = []
        #lx = []
        #ly = []
        u = []
        w = []
        for i, cell in enumerate(self.mesh.cells):
            #lx.append(str(cell.centroid[0]))
            #ly.append(str(cell.centroid[1]))
            location.append(' '.join(
                    [str(el) for el in cell.centroid])+' \n' )
            u.append(' '.join(
                    [ str(el) for el in self.u[cell.cid]])+' \n' )
            w.append(' '.join(
                    [str(el) for el in self.w[cell.cid]])+' \n' )
        FT.WriteLines(directory=self.solution_dir,
                      filename='cellcenters.dat',
                      lines = location)
        #conservative solution
        FT.WriteLines(directory=self.solution_dir,
                      filename='u_at_cellcenters.dat',
                      lines = u)
        #primative variables:
        FT.WriteLines(directory=self.solution_dir,
                      filename='w_at_cellcenters.dat',
                      lines = w)
        return
    
    def plot_flow_at_cell_centers(self, title):
        coords_ = []
        for i, cell in enumerate(self.mesh.cells):
            coords_.append(cell.centroid)
        coords_ = np.asarray(coords_)
        u_ = self.u
        w_ = self.w
        
        
        #--------------------------------------------------------------
        #
        # plot primative variables u,v
        Mc = np.sqrt(pow(w_[:,1], 2) + pow(w_[:,2], 2))
        #figure()
        fig, ax = plt.subplots()
        plt.title('Primative Variable Velocities')
        ax.axis('equal')
        # Q = quiver( coords_[:,0],coords_[:,1], 
        #            w_[:,0], w_[:,1], Mc, units='x', pivot='tip',width=.005, scale=3.3/.15)
        
        ax.quiver( coords_[:,0],coords_[:,1], 
                   w_[:,1], w_[:,2], Mc, units='x', pivot='tip',scale=1./15.)
        
        #--------------------------------------------------------------
        #
        # plot conservative u,v
        Mu = np.sqrt(pow(u_[:,1], 2) + pow(u_[:,2], 2))
        #figure()
        fig, ax = plt.subplots()
        plt.title('Conservative Variable Velocities')
        # Q = quiver( coords_[:,0],coords_[:,1], 
        #            u_[:,0], u_[:,1], Mu, units='x', pivot='tip',width=.005, scale=3.3/.15)
        
        ax.quiver( coords_[:,0],coords_[:,1], 
                   u_[:,1], u_[:,2], Mc, 
                   units='xy', angles='xy', pivot='tail',scale=1./15.)
        # plot conservative rho
        
        
        
        #--------------------------------------------------------------
        # plot density and pressure
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        
        
        #--------------------------------------------------------------
        # plot density
        
        # -----------------------
        # Interpolation on a grid
        # -----------------------
        # A contour plot of irregularly spaced data coordinates
        # via interpolation on a grid.
        
        # Create grid values first.
        npts = len(coords_)
        ngridx = self.mesh.m
        ngridy = self.mesh.n
        xi = np.linspace(self.mesh.xb, self.mesh.xe, ngridx)
        yi = np.linspace(self.mesh.yb, self.mesh.ye, ngridy)
        
        # Perform linear interpolation of the data (x,y)
        # on a grid defined by (xi,yi)
        triang = tri.Triangulation(coords_[:,0], coords_[:,1])
        interpolator = tri.LinearTriInterpolator(triang, u_[:,0])
        Xi, Yi = np.meshgrid(xi, yi)
        density = interpolator(Xi, Yi)
        
        # Note that scipy.interpolate provides means to interpolate data on a grid
        # as well. The following would be an alternative to the four lines above:
        #from scipy.interpolate import griddata
        #zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')
        
        ax1.contour(xi, yi, density, levels=14, linewidths=0.5, colors='k')
        #cntr1 = ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")
        cntr1 = ax1.contourf(xi, yi, density, cmap="RdBu_r")
        
        
        fig.colorbar(cntr1, ax=ax1)
        #ax1.plot(coords_[:,0], coords_[:,1], 'ko', ms=3)
        #ax1.set(xlim=(-2, 2), ylim=(-2, 2))
        ax1.set_title(title+' Density (%d points, %d grid points)' %
                      (npts, ngridx * ngridy))
        # 
        #--------------------------------------------------------------
        # plot pressure
        
        # Perform linear interpolation of the data (x,y)
        # on a grid defined by (xi,yi)
        #        triang = tri.Triangulation(coords_[:,0], coords_[:,1])
        #        interpolator = tri.LinearTriInterpolator(triang, u_[:,3])
        #        Xi, Yi = np.meshgrid(xi, yi)
        #        press = interpolator(Xi, Yi)
        
        # ----------
        # Tricontour
        # ----------
        # Directly supply the unordered, irregularly spaced coordinates
        # to tricontour.
        
        ax2.tricontour(coords_[:,0], coords_[:,1], u_[:,3], 
                       levels=14, linewidths=0.5, colors='k')
        cntr2 = ax2.tricontourf(coords_[:,0], coords_[:,1], u_[:,3], 
                                cmap="RdBu_r") #levels=14, cmap="RdBu_r")
        
        fig.colorbar(cntr2, ax=ax2)
        #ax2.plot(coords_[:,0], coords_[:,1], 'ko', ms=3)
        #ax2.set(xlim=(-2, 2), ylim=(-2, 2))
        ax2.set_title(title+' Pressure (%d points)' % npts)
        
        plt.subplots_adjust(hspace=0.5)
        plt.show()
        return
    
    
    def plot_flow_at_cell_centers_from_file(self):
        self.solution_dir = '../pics/solution'
        coords_ = np.loadtxt(self.solution_dir+'/cellcenters.dat')
        u_ = np.loadtxt(self.solution_dir+'/u_at_cellcenters.dat')
        w_ = np.loadtxt(self.solution_dir+'/w_at_cellcenters.dat')
        
        Mc = np.sqrt(pow(w_[:,0], 2) + pow(w_[:,0], 2))
        
        figure()
        Q = quiver( coords_[:,0],coords_[:,1], 
                   w_[:,0], w_[:,1], Mc, units='x', pivot='tip',width=.005, scale=3.3/.15)
        
        
        Mu = np.sqrt(pow(u_[:,0], 2) + pow(u_[:,0], 2))
        figure()
        Q = quiver( coords_[:,0],coords_[:,1], 
                   u_[:,0], u_[:,1], Mu, units='x', pivot='tip',width=.005, scale=3.3/.15)
        
        return
    
    def write_solution_to_vtk(self,filename_vtk=None):
        '''
        #*******************************************************************************
        # This subroutine writes a .vtk file for the grid whose name is defined by
        # filename_vtk.
        #
        # Use Paraview to read .vtk and visualize it.  https://www.paraview.org
        #
        #******************************************************************************
       
        '''
        if os.name == 'nt':
            self.solution_dir = '..\output\\vtk'
        else:
            self.solution_dir = '../output//vtk'
        zero = 0.0
        ntria = self.mesh.ntria
        nquad = self.mesh.nquad
        nnodes = self.mesh.nNodes
        dpn = self.dpn
        wn = np.zeros((nnodes,dpn),float)
        nc = np.zeros((nnodes),float)#<- nc(j) = # of cells contributing to node j.
        
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        # Compute solutions at nodes from solutions in cells to simplify visualization.
        #
        # Note: For 2nd-order scheme, this needs to be done with linear interpolation.
        # Note: Tecplot has an option to load cell-centered data.
        #------------------------------------------------------------------------------
        
        for i, cell in enumerate(self.mesh.cells):
            #Loop over vertices of the cell i
            for k, vtx in enumerate(cell.nodes):
                wn[vtx.nid,:] += self.w[i,:] #<- Add up solutions
                nc[vtx.nid] += 1. #<- Count # of contributing cells
        
        for i, vtx in enumerate(self.mesh.nodes):
            wn[vtx.nid,:] = wn[vtx.nid,:] / float(nc[vtx.nid])
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        print("\n\n-------------------------------------------------------\n")
        
        #------------------------------------------------------------------------------
        #header information
        
        lines = ['# vtk DataFile Version 3.0 \n']
        lines.append(filename_vtk+'\n')
        lines.append('ASCII'+'\n')
        lines.append('DATASET UNSTRUCTURED_GRID'+'\n')
        
        #------------------------------------------------------------------------------
        #nodal information
        lines.append('POINTS         '+ str(nnodes) + ' double'+'\n')
        for k, vtx in enumerate(self.mesh.nodes):
            lines.append(str(vtx.vector[0]) + ' ' 
                         +str(vtx.vector[1]) + ' ' +
                         str(zero)+'\n'
                         )
        
        #------------------------------------------------------------------------------
        #cell information
        lines.append('CELLS '+str(ntria+nquad) + ' ' +
                     str((3+1)*ntria + (4+1)*nquad)
                     +'\n'
                     )
        # Note: The latter is the number of integer values written below as data.
        #           4 for triangles (# of vertices + 3 vertices), and
        #           5 for quads     (# of vertices + 4 vertices).
        
        #---------------------------------
        # 2.1 List of triangles (counterclockwise vertex ordering)
        if ntria>0:
            # for i in range(ntria):
            #     lines.append('3', 
            #                  self.mesh.tria[i].nodes[0], 
            #                  self.mesh.tria[i].nodes[1], 
            #                  self.mesh.tria[i].nodes[2])
            for i in range(ntria):
                lines.append('3 '+ 
                             str(self.mesh.cells[i].nodes[0].nid) +' ' 
                             +str(self.mesh.cells[i].nodes[1].nid) +' '  
                             +str(self.mesh.cells[i].nodes[2].nid)
                             +'\n'
                             )
                
        
        #---------------------------------
        # 2.1 List of quads (counterclockwise vertex ordering)
        if nquad>0:
            # for i in range(nquad):
            #     lines.append('4', 
            #                  self.mesh.quad[i,0], 
            #                  self.mesh.quad[i,1], 
            #                  self.mesh.quad[i,2], 
            #                  self.mesh.quad[i,2])
            for i in range(nquad):
                lines.append('4 '+ 
                             str(self.mesh.cells[i].nodes[0].nid) +' ' 
                             +str(self.mesh.cells[i].nodes[1].nid) +' '  
                             +str(self.mesh.cells[i].nodes[2].nid) +' ' 
                             +str(self.mesh.cells[i].nodes[3].nid)
                             +'\n'
                             )
                
        
        #---------------------------------
        # Cell type information.
        lines.append('CELL_TYPES  ' + str(ntria+nquad)+'\n')
        
        # Triangle is classified as the cell type 5 in the .vtk format.
        if ntria>0:
            for i in range(ntria):
                lines.append('5'+'\n')
        # quad is classified as the cell type 9 in the .vtk format.
        if nquad>0:
            for i in range(nquad):
                lines.append('9'+'\n')
                
        
        #---------------------------------
        # field data (density, pressure, velocity)
        lines.append('POINT_DATA  '+ str(nnodes)+'\n')
        lines.append('FIELD  FlowField  4'+'\n')
        
        lines.append('Density    1 ' + str(nnodes) + '  double'+'\n')
        for i in range(nnodes):
            lines.append(str(wn[i,0])+'\n')
        
        lines.append('X-velocity    1 ' + str(nnodes) + '  double'+'\n')
        for i in range(nnodes):
            lines.append(str(wn[i,1])+'\n')
        
        lines.append('Y-velocity    1 ' + str(nnodes) + '  double'+'\n')
        for i in range(nnodes):
            lines.append(str(wn[i,2])+'\n')
        
        lines.append('Pressure    1 ' + str(nnodes) + '  double'+'\n')
        for i in range(nnodes):
            lines.append(str(wn[i,3])+'\n')
            
        
        #vector data
        
        lines.append('\nVECTORS Velocity double \n')
        for i in range(nnodes):
            lines.append(str(wn[i,1])+' ' +str(wn[i,2])+' 0.0 \n')
        
        
        FT.WriteLines(directory=self.solution_dir,
                      filename=filename_vtk,
                      lines = lines)
        
        print(' End of Writing .vtk file = ' + filename_vtk)
        print("-------------------------------------------------------")
        #---------------------------------------------------------------------------
        
        return
# def linewriter_array(inistr, data):
#     lend = len(data)
#     for i, el in enumerate(data):
#         inistr.append(str(el))
#     return
# def linewriter_float(inistr, data):
    
#     return
    
    
class FlowState(object):
    
    def __init__(self, rho_inf=1., u_inf=1., v_inf=1., p_inf=1.):
        self.rho_inf = rho_inf
        self.u_inf =  u_inf
        self.v_inf = v_inf
        self.p_inf = p_inf
        return
    
def show_LSQ_grad_area_plots(solver):
    for cc in solver.cclsq[55:60]:
        cc.plot_lsq_reconstruction()
    return

def show_one_tri_cell(solver):
    cc = solver.cclsq[57]
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
        #uplevel = os.path.join(os.path.dirname(__file__), '..','cases')
        uplevel = os.path.join(os.path.dirname(os.getcwd()), 'cases')
        #path2vortex = uplevel+'\\cases\case_unsteady_vortex'
        path2case = os.path.join(uplevel, 'case_unsteady_vortex')
        self.DHandler = DataHandler(project_name = 'vortex',
                                       path_to_inputs_folder = path2case)
        
        
        self.grid = Grid(generated=False,
                         dhandle = self.DHandler,
                         type_='tri',
                         winding='ccw')
        
        

        
class TestSteadyCylinder(object):
    
    def __init__(self):
        # up a level
        #uplevel = os.path.join(os.path.dirname(__file__), '..','cases')
        uplevel = os.path.join(os.path.dirname(os.getcwd()), 'cases')
        #path2vortex = uplevel+'\\cases\case_unsteady_vortex'
        path2case = os.path.join(uplevel, 'case_steady_cylinder')
        self.DHandler = DataHandler(project_name = 'cylinder',
                                       path_to_inputs_folder = path2case)
        
        
        self.grid = Grid(generated=False,
                         dhandle = self.DHandler,
                         type_='tri',
                         winding='ccw')
        
        
class TestSteadyAirfoil(object):
    
    def __init__(self):
        # up a level
        #uplevel = os.path.join(os.path.dirname(__file__), '..','cases')
        uplevel = os.path.join(os.path.dirname(os.getcwd()), 'cases')
        #path2vortex = uplevel+'\\cases\case_unsteady_vortex'
        path2case = os.path.join(uplevel, 'case_steady_airfoil')
        self.DHandler = DataHandler(project_name = 'airfoil',
                                       path_to_inputs_folder = path2case)
        
        
        self.grid = Grid(generated=False,
                         dhandle = self.DHandler,
                         type_='quad',
                         winding='ccw')

    
    
class TestTEgrid(object):
    
    def __init__(self):
        # up a level
        #uplevel = os.path.join(os.path.dirname(__file__), '..','cases')
        uplevel = os.path.join(os.path.dirname(os.getcwd()), 'cases')
        #path2vortex = uplevel+'\\cases\case_unsteady_vortex'
        path2case = os.path.join(uplevel, 'case_verification_te')
        self.DHandler = DataHandler(project_name = 'test',
                                       path_to_inputs_folder = path2case)
        
        
        self.grid = Grid(generated=False,
                         dhandle = self.DHandler,
                         type_='quad',
                         winding='ccw')
        

class TestShockDiffractiongrid(object):
    
    def __init__(self):
        # up a level
        uplevel = os.path.join(os.path.dirname(os.getcwd()), 'cases')
        path2case = os.path.join(uplevel, 'case_shock_diffraction')
        self.DHandler = DataHandler(project_name = 'shock',
                                       path_to_inputs_folder = path2case)
        
        
        self.grid = Grid(generated=False,
                         dhandle = self.DHandler,
                         type_='quad',
                         winding='ccw')
    
        
        
# class TestQEgrid(object):
    
#     def __init__(self):
#         # up a level
#         #uplevel = os.path.join(os.path.dirname(__file__), '..','cases')
#         uplevel = os.path.join(os.path.dirname(os.getcwd()), 'cases')
#         #path2vortex = uplevel+'\\cases\case_unsteady_vortex'
#         path2vortex = os.path.join(uplevel, 'case_verification_te')
#         self.DHandler = DataHandler(project_name = 'te_test',
#                                         path_to_inputs_folder = path2vortex)
        
        
#         self.grid = Grid(generated=False,
#                           dhandle = self.DHandler,
#                           type_='quad',
#                           winding='ccw')
    

if __name__ == '__main__':
    # gd = Grid(type_='quad',m=10,n=10,
    #           winding='ccw')
    
    #mesh = Grid(type_='tri',m=42,n=21,
    #          winding='ccw')
    
    #mesh = Grid(type_='quad',m=42,n=21,
    #          winding='ccw')
    #mesh = Grid(generated=True,type_='quad',m=42,n=21,
    #          winding='ccw')
    
    #cell = mesh.cellList[44]
    #face = cell.faces[0]
    
    #cell.plot_cell()
    
    vtkNames = {0:'vortex.vtk',
                1:'airfoil.vtk',
                2:'cylinder.vtk',
                3:'test.vtk',
                4:'shock_diffraction.vtk'}
    
    thisTest = 0
    whichTest = {0:TestInviscidVortex,
                 1:TestSteadyAirfoil,
                 2:TestSteadyCylinder,
                 3:TestTEgrid,
                 4:TestShockDiffractiongrid}
    #test = TestInviscidVortex()
    #test = TestSteadyAirfoil()
    #test = TestSteadyCylinder()
    #test = TestTEgrid()
    #test = TestShockDiffractiongrid()
    test = whichTest[thisTest]()
    
    
    #if False:
    if True:
        
        #'''
        self = Solvers(mesh = test.grid)
        
        #cc = self.cclsq[35]
        #cc.plot_lsq_reconstruction()
        
        
        #----------------------------
        # plot LSQ gradient stencils
        #show_LSQ_grad_area_plots(self)
        
        
        # cc = self.cclsq[57]
        # cc.plot_lsq_reconstruction()
        # cell = cc.cell
        # cell.plot_cell() #normals should be outward facing
        
        #'''
        
        #"""
        whichSolver = {0: 'vortex',
                       1: 'freestream',
                       2: 'freestream',
                       3: 'mms',
                       4:'shock-diffraction'}
        #self.solver_boot(flowtype = 'mms') #TODO fixme compute_manufactured_sol_and_f_euler return vals
        #self.solver_boot(flowtype = 'freestream')
        #self.solver_boot(flowtype = 'vortex')
        #self.solver_boot(flowtype = 'shock-diffraction')
        
        self.solver_boot(flowtype = whichSolver[thisTest])
        
        #self.plot_flow_at_cell_centers(title = 'Initial Solution')
        
        self.write_solution_to_vtk('init_'+vtkNames[thisTest])
        
        solvertype = {0:'explicit_unsteady_solver',
                      1:'explicit_steady_solver',
                      2:'explicit_steady_solver',
                      3:'mms_solver',
                      4:'explicit_unsteady_solver_efficient_shockdiffraction'}
        #'''
        self.solver_solve( tfinal=10., dt=.01, 
                          solver_type = solvertype[thisTest])
        
        self.write_solution_to_vtk(vtkNames[thisTest])
        #'''
        ################################
        '''
        self.solver_solve( tfinal=0.2, dt=.01, 
                           solver_type = solvertype[1])
        print ('nfaces :',len(self.mesh.faceList))
        #self.write_solution_to_vtk('test.vtk')
        self.write_solution_to_vtk(vtkNames[thisTest])
        #'''
        ################################
        '''
        self.solver_solve( tfinal=0.2, dt=.01, 
                           solver_type = solvertype[2])
        #'''
        
        '''
        self.plot_solution( title='Final ')
        #'''
        
    
        # print('--------------------------------')
        # print('validate normals on boundaries')
        # for bound in self.mesh.bound:
        #     print(bound.bc_type)
        # for face in self.mesh.boundaryList:
        #     print(face.compute_normal(True))
    
        
        '''
        # if memory issues are encountered:
        del(self)
        del(mesh)
        
        canvas = plotmesh = PlotGrid(self.mesh)
        plotmesh.plot_boundary() #normals should be outward facing
        
        for bface in self.mesh.boundaryList:
            print(bface.parentcell.cid,bface.face_nrml_mag)
        
        plotmesh = PlotGrid(self.mesh)
        axTri = plotmesh.plot_cells()
        axTri = plotmesh.plot_centroids(axTri)
        axTri = plotmesh.plot_face_centers(axTri)
        axTri = plotmesh.plot_normals(axTri)
        
        plotmesh = PlotGrid(self.mesh)
        axRect = plotmesh.plot_cells()
        axRect = plotmesh.plot_centroids(axRect)
        axRect = plotmesh.plot_face_centers(axRect)
        axRect = plotmesh.plot_normals(axRect)
        
        
        #'''
        self.print_nml_data()