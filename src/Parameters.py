#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 00:17:21 2020

@author: lukemcculloch
"""

class Parameters(object):
    '''
    todo: handle defaults better
    '''
    
    def __init__(self,iparams):
        self.iparams = iparams
        self.get_parameters()
        
        
    def get_parameters(self):
        
        iparams = self.iparams
        
        
        if iparams['aoa'] == False:
            self.aoa = 0.0
        else:
            self.aoa = float(iparams['aoa'])
            
            
        if iparams['M_inf'] == False:
            self.M_inf = 0.3
        else:
            self.M_inf = float(iparams['M_inf'])
            
            
        if iparams['CFL'] == False:
            self.CFL = 0.9
        else:
            self.CFL = float(iparams['CFL'])
            
        if iparams['inviscid_flux'] == False:
            self.inviscid_flux = 'roe'
        else:
            self.inviscid_flux = iparams['inviscid_flux']
            
            
        if iparams['second_order'] == False:
            self.second_order = True #second_order default because second_order is not listed (see todos because this is confusing)
        else:
            #self.second_order = iparams['second_order']
            self.second_order = (True if iparams['second_order']=='T' else False)
            
            
        if iparams['eig_limiting_factor'] == False:
            self.eig_limiting_factor = 0.0
        else:
            self.eig_limiting_factor = iparams['eig_limiting_factor']
            
            
        if iparams['compute_te_mms'] == False:
            self.compute_te_mms = 'F'
        else:
            #self.compute_te_mms = iparams['compute_te_mms']
            self.compute_te_mms = (True if iparams['compute_te_mms']=='T' else False)
            
            
        if iparams['use_limiter'] == False: #limiter is not mentioned
            self.use_limiter = False #the default appears to be false, as per mme test
        else:
            self.use_limiter = (True if iparams['use_limiter']=='T' else False)
            
        
        if iparams['do_amr'] == False:
            self.do_amr = False
        else:
            self.do_amr = bool(iparams['do_amr'])
            
        if iparams['refine_threshold'] == False:
            self.refine_threshold = 0.2
        else:
            self.refine_threshold = float(iparams['refine_threshold']) #namelist.get('refine_threshold', 0.2)
        
        if iparams['coarsen_threshold'] == False:
            self.coarsen_threshold = 0.1
        else:
            self.coarsen_threshold = float(iparams['coarsen_threshold']) #namelist.get('coarsen_threshold', 0.1)
            
            
        