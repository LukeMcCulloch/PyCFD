#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 00:36:00 2020

@author: lukemcculloch
"""
from __future__ import print_function
import numpy as np
class BC_states(object):
    """Boundary Conditions (BC)
    """
    
    def __init__(self, solver, flowstate, p2):
        self.solver = solver
        self.flowstate = flowstate
        
    
    
    def get_right_state(self, xb,yb,ucL,njk,bc_state_type, ucb):
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
        wL = np.zeros(4,float)
        wb = np.zeros_like(wL)
        dummy =  np.zeros_like(wL)
        
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
                    'outflow_supersonic':[wL, wb],
                    'dirichlet':[]
                    }
        
        getattr(self, bc_state_type)(*vs_cases[bc_state_type])
        
        #Dirichlet assumes the manufactured solution: so, compute wb for (xb,yb)
        #compute_manufactured_sol_and_f_euler(xb,yb, wb,dummy)
        
        
        
        return
    

    #**************************************************************************
    # Default
    #**************************************************************************
    def default(self):
        print( "Boundary condition=",self.bc_state_type,"  not implemented." )
        print( " --- Stop at get_right_state() in solver B.C.s" )
        return
    
    
    
    #**************************************************************************
    # Freestream
    #**************************************************************************
    def freestream(self, wb):
        flowstate = self.flowstate
        wb[0] = flowstate.rho_inf
        wb[1] = flowstate.u_inf
        wb[2] = flowstate.v_inf
        wb[3] = flowstate.p_inf
        return
    
    #**************************************************************************
    # Outflow supersonic
    #**************************************************************************
    @staticmethod
    def outflow_supersonic(self):
        return