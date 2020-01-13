#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:15:20 2020

@author: lukemcculloch
"""
import numpy as np
Array = np.zeros
sqrt = np.sqrt


#-----------------------------------------------------------------------------#
# Euler solver: Explicit Unsteady Solver: Ut + Fx + Gy = S
#
# This subroutine solves an un steady problem by 2nd-order TVD-RK with a
# global time step.
#-----------------------------------------------------------------------------#
def explicit_steady_solver():
    return 