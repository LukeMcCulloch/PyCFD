#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 22:55:46 2020

@author: lukemcculloch
"""

from FileTools import GetLines, GetLineByLine


class DataHandler(object):
    
    def __init__(self, project_name, path_to_inputs_folder):
        
        self.GetLines = GetLines
        
        self.path_to_inputs_folder = path_to_inputs_folder
        
        
        #----------------------------------------------------------
        # Input parameters file
        self.filename_nml = 'input.nml'
        self.makeDictionary()
        self.readinput()
        project_name = self.inputParameters['project_name']
        
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
        
        #for key in self.inputParameters.keys():
        #    val = self.inputParameters[key]
        #    #print(key, ' , ', val)
        #    if val == 'T':
        #        self.inputParameters[key] = True
                
                
            
        
        
        print(" End of file names setup..... ")
    
    def makeDictionary(self):
        '''
        todo: make dictionary completely on the fly 
        handle all else with defaults in the solver parameters
        '''
        
        self.inputParameters = {'project_name'          : False,
                              'steady_or_unsteady'      : False,
                              't_final'                 : False,
                              'generate_tec_file'       : False,
                              'generate_vtk_file'       : False,
                              'M_inf'                   : False,
                              'aoa'                     : False,
                              'inviscid_flux'           : False,
                              'eig_limiting_factor'     : False,
                              'CFL'                     : False,
                              'second_order'            : False,
                              'first_order'             : False,
                              'use_limiter'             : False,
                              'compute_te_mms'          : False,
                              'do_amr'                  : False,
                              'refine_threshold'        : False,
                              'coarsen_threshold'       : False}
    
    def readinput(self):
        self.ilines = GetLines(directory = self.path_to_inputs_folder,
                               filename = self.filename_nml)
        for line in self.ilines:
            tokens = line.split()
            if len(tokens) == 3:
                self.inputParameters[tokens[0]] = tokens[2]
            elif len(tokens) > 3:
                self.inputParameters[tokens[0]] = []
                newtokens = line.split(',')
                for tok in newtokens[2:]:
                    self.inputParameters[tokens[0]].append(float(tok))
        return