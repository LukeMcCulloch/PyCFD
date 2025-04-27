#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 17:41:31 2025

@author: lukemcculloch
"""



vtkNames = {0:'vortex.vtk',
            1:'airfoil.vtk',
            2:'cylinder.vtk',
            3:'test.vtk',
            4:'shock_diffraction.vtk'}

whichSolver = {0: 'vortex',
               1: 'freestream',
               2: 'freestream',
               3: 'mms',
               4:'shock-diffraction'}

solvertype = {0:'explicit_unsteady_solver',
              1:'explicit_steady_solver',
              2:'explicit_steady_solver',
              3:'mms_solver',
              4:'explicit_unsteady_solver_efficient_shockdiffraction'}

