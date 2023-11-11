#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 00:36:00 2020

@author: lukemcculloch
"""
from __future__ import print_function
#
import weakref
#
import numpy as np
#
from ManufacturedSolutions import compute_manufactured_sol_and_f_euler


class BC_states(object):
    """
        Boundary Conditions (BC)
    """
    
    def __init__(self, solver, flowstate):
        #self.solver = solver
        self._solver =  weakref.ref(solver) if solver else solver
        self.flowstate = flowstate
        
    @property
    def solver(self):
        if not self._solver:
            return self._solver
        _solver = self._solver()
        if _solver:
            return _solver
        else:
            raise LookupError("solver was destroyed")
            
    def __del__(self):
        print("delete", 'BC_state')
        
    
    
    def get_right_state(self, 
                        xb, yb,
                        ucL, njk,
                        bc_state_type,
                        f):
        """
        # ---------------------------------------------------------------------
        #
        # ---------------------------------------------------------------------
        #  Input: xb,yb = boundary face midpoint
        #           ucL = Interior conservative variables (rho, rho*u, rho*v, rho*E)
        #           njk = Outward boundary normal vector.
        #           bc_state_type = BC name
        # 
        # Output:   ucb = Boundary state in conservative variables (rho, rho*u, rho*v, rho*E)
        # ---------------------------------------------------------------------
        #
        # Note:    E = p/(gamma-1)/rho + 0.5*(u^2+v^2)
        #       -> p = (gamma-1)*rho*E-0.5*rho*(u^2+v^2)
        #
        #
        

        #Input
         float                  :: xb, yb
         array((4,1),float)     :: ucL
         array((2,1),float)     :: njk
         string                 :: bc_state_type
        
        #Output
         array((4,1),float)     :: ucb
        
        """
        
        solver = self.solver
        #Local variables
        #wL = np.zeros(4,float)
        #wb = np.copy(ucb) #
        wb = np.zeros_like(ucL)
        #dummy =  np.zeros_like(wL)
        
        #---------------------------------------------------------
        # Get the primitive variables [rho,u,v,p] as input to
        # the following subroutines, which return the boundary
        # state in the primitive variables.
        wL = solver.u2w(ucL)

        #---------------------------------------------------------
        # Below, input is wLp = the primitive variabes [rho,u,v,p].
        # Output is the right state in wRp = [rho,u,v,p].
        
        vs_cases = {'freestream':[wb],
                    'outflow_subsonic':[wL, wb],
                    'symmetry_y':[wL,njk,wb],
                    'slip_wall':[wL,njk, wb],
                    'slip_wall_ymmtm_fix':[wL,njk, wb],
                    'outflow_supersonic':[wL, wb],
                    'dirichlet':[xb,yb,f[:]]
                    }
        
        #print('before:  wL,njk, wb',wL,njk, wb)
        
        if bc_state_type == 'dirichlet':
            wb[:], f[:] = getattr(self, bc_state_type)(*vs_cases[bc_state_type])
        else:
            wb[:] = getattr(self, bc_state_type)(*vs_cases[bc_state_type])

        #print('after: wL,njk, wb',wL,njk, wb)
        
        
        #---------------------------------------------------------
        # Return the right state in conservative variables:
        #                                 [rho,rho*u,rho*v,rho*E]
        ucb = self.solver.w2u(wb)
        #print('wb',wb)
        #print('ucb',ucb)
        return ucb
    

    #**************************************************************************
    # Default
    #**************************************************************************
    def default(self):
        print( "Boundary condition=",self.bc_state_type,"  not implemented." )
        print( " --- Stop at get_right_state() in solver B.C.s" )
        return
    
    
    
    #**************************************************************************
    # Dirichlet
    #**************************************************************************
    def dirichlet(self, xb, yb, f):
        #print("dirichlet, mms")
        #print( 'got wb',wb)
        # flowstate = self.flowstate
        # wb[1] = 0.0
        # wb[2] = 0.0
        # #print( 'set wb',wb)
        #Dirichlet assumes the manufactured solution: so, compute wb for (xb,yb)
        return compute_manufactured_sol_and_f_euler(xb,yb,f[:]) #TODO: fix return!
    
    
    #**************************************************************************
    # Freestream
    #**************************************************************************
    def freestream(self, wb):
        #print("freestream")
        #print( 'got wb',wb)
        flowstate = self.flowstate
        wb[0] = flowstate.rho_inf
        wb[1] = flowstate.u_inf
        wb[2] = flowstate.v_inf
        wb[3] = flowstate.p_inf
        #print( 'set wb',wb)
        return wb
    
    
    
    #**************************************************************************
    # Subsonic outflow (backpressure)
    #**************************************************************************
    def outflow_subsonic(self, *args):
        #print("back-pressure")
        wL = args[0] 
        wb = args[1]
        #print('wL',wL)
        #print('wb',wb)
        flowstate = self.flowstate
        #-------------------------
        # Back pressure condition
        #wb = np.copy(wL)
        wb[:] = wL[:]
        wb[3] = flowstate.p_inf  #<- fix the pressure
        #print('wL',wL)
        #print('wb',wb)
        return wb


    #**********************************************************************
    # Symmetry w.r.t. x-axis, which is called y-symmetry here.
    #
    # Note: This is a simplified implementation similar to slip wall condition.
    #**********************************************************************
    def symmetry_y(self, *args):
        #print("symmetry_y")
        wL = args[0]
        njk = args[1]
        wb = args[2]
        
        
        un = wL[1]*njk[0] + wL[2]*njk[1]  #not used

        #-------------------------
        # Define the right state:
        
        #wb = np.copy(wL)
        wb[:] = wL[:]
        
        #-------------------------
        # Ensure zero y-momentum flux on average:
        
        #wb[2] = 0.0 #where did this come from??
        
        # Ensure zero normal velocity on average:
        
        wb[1] = wL[1] - 2.0*un*njk[0]
        wb[2] = wL[2] - 2.0*un*njk[1]
        
        # (ub,vb) = (uL,vL) - 2*un*njk -> 0.5[(ub,vb)+(uL,vL)]*njk = (0,0).
        # Since rho_b = rhoL as set in the above, this means the momemtum
        # in n direction is also zero.
        #print('wb = {}'.format(wb))
        return wb

 
    #**************************************************************************
    # Slip wall
    #
    #**************************************************************************
    def slip_wall(self, wL,njk, wb):
        #print("slip_wall")

        un = wL[1]*njk[0] + wL[2]*njk[1]
        #print('un = ',un)
        
        #-------------------------
        # Define the right state:
        
        wb[:] = wL[:]
        
        # Ensure zero normal velocity on average:
        
        wb[1] = wL[1] - un*njk[0]
        wb[2] = wL[2] - un*njk[1]
        
        #print('wb dot njk= {}'.format(wb[1:3].dot(njk)))
        #tol= 1.e-10
        #assert(wb[1:3].dot(njk) < tol),"slip wall failure normal velocity = {}".format(wb[1:3].dot(njk))
        return wb
    
    
    #**************************************************************************
    # Outflow supersonic
    #**************************************************************************
    def outflow_supersonic(self, wL, wb):
        """
            wb = np.array(4, float)
            wL = np.array(4, float)
        """
        #print("outflow_supersonic")
        #---------------------------------------------
        # Take everything from the interior.
        
        wb[:] = wL[:]
        
        return wb
    
    
    #**************************************************************************
    # Special Shock diffraction
    #**************************************************************************
    def slip_wall_ymmtm_fix(self, wL,njk, wb):
        #print("sock diffraction slip_wall y-mmtm fix")

        un = wL[1]*njk[0] + wL[2]*njk[1]
        #print('un = ',un)
        
        #-------------------------
        # Define the right state:
        
        wb[:] = wL[:] 
        #wb[2] = 0.0 #tlm todo: understand this special bc
        
        # Ensure zero normal velocity on average:
        
        wb[1] = wL[1] - un*njk[0]
        wb[2] = wL[2] - un*njk[1]
        
        
        #wb[2] = 0.0 #tlm todo: understand this special bc
        
        #print('wb dot njk= {}'.format(wb[1:3].dot(njk)))
        #tol= 1.e-10
        #assert(wb[1:3].dot(njk) < tol),"slip wall failure normal velocity = {}".format(wb[1:3].dot(njk))
        return wb
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------