#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 00:36:00 2020

@author: lukemcculloch
"""

class BC_states(object):
    
    def __init__(self, p2):
        self.p2 = p2
    
    
    def get_right_state(self):
        """
        #
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
        """
        return
    
    
    
    def outflow_supersonic(self):
        return