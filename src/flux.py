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
# Riemann solver: Roe's approximate Riemann solver
#-----------------------------------------------------------------------------#
def roe(nx,gamma,uL,uR,f,fL,fR) :
    dd = Array((3),float)
    dF = Array((3),float)
    V = Array((3),float)
    gm = gamma-1.0

    #for i = 1:nx+1
    for i in range(1,nx+1):
        #Left state:
        rhLL = uL[i,0]
        uuLL = uL[i,1]/rhLL
        eeLL = uL[i,2]/rhLL
        ppLL = gm*(eeLL*rhLL - 0.5*rhLL*(uuLL*uuLL))
        hhLL = eeLL + ppLL/rhLL
        
        #right state:
        rhRR = uR[i,0]
        uuRR = uR[i,1]/rhRR
        eeRR = uR[i,2]/rhRR
        ppRR = gm*(eeRR*rhRR - 0.5*rhRR*(uuRR*uuRR))
        hhRR = eeRR + ppRR/rhRR
        
        alpha = 1.0/(sqrt(abs(rhLL)) + sqrt(abs(rhRR)))
        uu = (sqrt(abs(rhLL))*uuLL + sqrt(abs(rhRR))*uuRR)*alpha
        hh = (sqrt(abs(rhLL))*hhLL + sqrt(abs(rhRR))*hhRR)*alpha
        aa = sqrt(abs(gm*(hh-0.5*uu*uu)))
        
        D11 = abs(uu)
        D22 = abs(uu + aa)
        D33 = abs(uu - aa)
        
        beta = 0.5/(aa*aa)
        phi2 = 0.5*gm*uu*uu
        
        #Right eigenvector matrix
        R11, R21, R31 = 1.0, uu, phi2/gm
        R12, R22, R32 = beta, beta*(uu + aa), beta*(hh + uu*aa)
        R13, R23, R33 = beta, beta*(uu - aa), beta*(hh - uu*aa)

        #Left eigenvector matrix
        L11, L12, L13 = 1.0-phi2/(aa*aa), gm*uu/(aa*aa), -gm/(aa*aa)
        L21, L22, L23 = phi2 - uu*aa, aa - gm*uu, gm
        L31, L32, L33 = phi2 + uu*aa, -aa - gm*uu, gm

        for m in range(3):
			V[m] = 0.5*(uR[i,m]-uL[i,m])
            

        dd[0] = D11*(L11*V[0] + L12*V[1] + L13*V[2])
        dd[1] = D22*(L21*V[0] + L22*V[1] + L23*V[2])
        dd[2] = D33*(L31*V[0] + L32*V[1] + L33*V[2])
        
        dF[0] = R11*dd[0] + R12*dd[1] + R13*dd[2]
        dF[1] = R21*dd[0] + R22*dd[1] + R23*dd[2]
        dF[2] = R31*dd[0] + R32*dd[1] + R33*dd[2]


        for m in range(3):
            f[i,m] = 0.5*(fR[i,m]+fL[i,m]) - dF[m]
            
    
    return 