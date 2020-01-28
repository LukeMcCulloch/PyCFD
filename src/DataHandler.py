#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 22:55:46 2020

@author: lukemcculloch
"""

from FileTools import GetLines


def DataHandler(object):
    
    def __init__(self, project_name, path_to_inputs_folder):
        
        self.path_to_inputs_folder = path_to_inputs_folder
        
        #----------------------------------------------------------
        # Input grid file (.ugrid):
        self.filename_grid = project_name + ".grid"
        
        #----------------------------------------------------------
        # Input boundary condition file (ASCII file)
        self.filename_bc = project_name + ".bc"
        
        #----------------------------------------------------------
        # Output: plot file (ASCII file)
        self.filename_plot = project_name + "_plot.dat"
        
        #----------------------------------------------------------
        # Output: .vtk file (ASCII file)
        self.filename_vtk = project_name + ".vtk"
        
        #----------------------------------------------------------
        # Output: plot file (ASCII file)
        self.filename_plot_hist = project_name + "._plot_hist.dat"
        
        print(" End of file names setup..... ")
        
        pass