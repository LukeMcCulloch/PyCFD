#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 00:36:00 2020

@author: lukemcculloch
"""
import numpy as np

class BC_states(object):
    
    def __init__(self, p2):
        self.p2 = p2
    
    
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
        #Local variables
        wL = np.zeros(4,float)
        wb = np.zeros_like(wL)
        dummy =  np.zeros_like(wL)
        
        #---------------------------------------------------------
        # Get the primitive variables [rho,u,v,p] as input to
        # the following subroutines, which return the boundary
        # state in the primitive variables.
        
        wL = u2w(ucL)
        
        return
    
    
    
    def outflow_supersonic(self):
        return