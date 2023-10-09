# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 07:38:06 2023

@author: LUKE.MCCULLOCH
"""

class dbInterfaceFlux(object):
                                        
    def __init__(self, u1, u2,                  #<- Left/right states
                    gradw1, gradw2,             #<- Left/right same gradients
                    face,                       #<- unit face normal
                    c1,                         #<- Left cell centroid
                    c2,                         #<- right cell centroid
                    xm, ym,                     #<- face midpoint
                    phi1, phi2                  #<- Limiter functions
                    ):
                    
        self.u1 = u1
        self.u2 = u2
        self.gradw1 = gradw1
        self.gradw2 = gradw2
        self.face = face
        self.c1 = c1
        self.c2 = c2
        self.xm = xm
        self.ym = ym
        self.phi1 = phi1
        self.phi2 = phi2
        
    
    def __str__(self):
        print('u1 = self.dbugIF.u1')
        print('u2 = self.dbugIF.u2')
        #print('gradw1 = self.dbugIF.gradw1')
        #print('gradw2 = self.dbugIF.gradw2')
        print('face = self.dbugIF.face')
        print('c1 = self.dbugIF.c1')
        print('c2 = self.dbugIF.c2')
        print('xm = self.dbugIF.xm')
        print('ym = self.dbugIF.ym')
        print('phi1 = self.dbugIF.phi1')
        print('phi2 = self.dbugIF.phi2')
        
        print('\ngradw1 = self.dbugIF.gradw1')
        print('gradw2 = self.dbugIF.gradw2\n')
        
        print('n12 =   self.dbugIF.face.normal_vector')              # Directed area vector (unit vector)
        print('C1  = c1.centroid')          # left centroid
        print('C2  = c2')         # right centroid for boundary, interior needs.centroid!
        return '#dbugIF data'
    
    
    
    

class dbRoeFlux(object):
    
    def __init__(self,ucL, ucR, njk, num_flux, wsn, gamma = 1.4):
        self.ucL = ucL
        self.ucR = ucR
        self.njk = njk
        self.num_flux = num_flux
        self.wsn = wsn
        self.gamma = gamma
        
        
    def __str__(self):
        print('ucL = self.dbRoeFlux.ucL')
        print('ucR = self.dbRoeFlux.ucR')
        print('njk = self.dbRoeFlux.njk')
        print('num_flux = self.dbRoeFlux.num_flux')
        print('wsn = self.dbRoeFlux.wsn')
        print('gamma = self.dbRoeFlux.gamma')
        return '#dbRoeFlux data'