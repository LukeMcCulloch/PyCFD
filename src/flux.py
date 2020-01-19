#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:15:20 2020

"""
import numpy as np
Array = np.zeros
sqrt = np.sqrt



#********************************************************************************
#* -- 3D Roe's Flux Function and Jacobian --
#*
#* NOTE: This version does not use any tangent vector.
#*       See "I do like CFD, VOL.1" about how tangent vectors are eliminated.
#*
#* This subroutine computes the Roe flux for the Euler equations
#* in the direction, njk=[nx,ny,nz].
#*
#* P. L. Roe, Approximate Riemann Solvers, Parameter Vectors and Difference
#* Schemes, Journal of Computational Physics, 43, pp. 357-372.
#*
#* Conservative form of the Euler equations:
#*
#*     dU/dt + dF/dx + dG/dy + dH/dz = 0
#*
#* This subroutine computes the numerical flux for the flux in the direction,
#* njk=[nx,ny,nz]:
#*
#*     Fn = F*nx + G*ny + H*nz = | rho*qn          |
#*                               | rho*qn*u + p*nx |
#*                               | rho*qn*v + p*ny |
#*                               | rho*qn*w + p*nz |
#*                               | rho*qn*H        |    (qn = u*nx + v*ny + w*nz)
#*
#* The Roe flux is implemented in the following form:
#*
#*   Numerical flux = 1/2 [ Fn(UR) + Fn(UL) - |An|dU ], 
#*
#*  where
#*
#*    An = dFn/dU,  |An| = R|Lambda|L, dU = UR - UL.
#*
#* The dissipation term, |An|dU, is actually computed as
#*
#*     sum_{k=1,4} |lambda_k| * (LdU)_k * r_k,
#*
#* where lambda_k is the k-th eigenvalue, (LdU)_k is the k-th wave strength,
#* and r_k is the k-th right-eigenvector evaluated at the Roe-average state.
#*
#* Note: The 4th component is a combined contribution from two shear waves.
#*       They are combined to eliminate the tangent vectors.
#*       So, (LdU)_4 is not really a wave strength, and
#*       r_4 is not really an eigenvector.
#*       See "I do like CFD, VOL.1" about how tangent vectors are eliminated.
#*
#* Note: In the code, the vector of conserative variables are denoted by uc.
#*
#* ------------------------------------------------------------------------------
#*  Input: ucL(1:5) =  Left state (rhoL, rhoL*uL, rhoL*vL, rhoL*wR, rhoL*EL)
#*         ucR(1:5) = Right state (rhoR, rhoL*uR, rhoL*vR, rhoL*wR, rhoL*ER)
#*         njk(1:3) = unit face normal vector (nx, ny, nz), pointing from Left to Right.
#*
#*           njk
#*  Face normal ^   o Right data point
#*              |  .
#*              | .
#*              |. 
#*       -------x-------- Face
#*             .                 Left and right states are
#*            .                   1. Values at data points for 1st-order accuracy
#*           .                    2. Extrapolated values at the face midpoint 'x'
#*          o Left data point        for 2nd/higher-order accuracy.
#*
#*
#* Output:  num_flux(1:5) = the numerical flux vector
#*                    wsn = maximum wave speed (eigenvalue)
#*
#* ------------------------------------------------------------------------------
#*
#* Note: This subroutine has been prepared for an educational purpose.
#*       It is not at all efficient. Think about how you can optimize it.
#*       One way to make it efficient is to reduce the number of local variables,
#*       by re-using temporary variables as many times as possible.
#*
#* Note: Please let me know if you find bugs. I'll greatly appreciate it and
#*       fix the bugs.
#*
#* Katate Masatsuka, November 2012. http://www.cfdbooks.com
#*
#* converted to python by Luke McCulloch
#*
#********************************************************************************
def roe(ucL, ucR, njk, num_flux,wsn):
    """
    3D Roe approximate Riemann Solver for 
    the flux across a face
    
    input:
    ucL = np.zeros(5,float)         #conservative variables in 3D
    ucR = np.zeros(5,float)         #conservative variables in 3D
    njk = np.zeros(3,float)         #face normal in 3D
    
    output:
    num_flux                        #Numerical viscous flux
    wsn                             # max wave speed
    """
    
    #Some constants
    zero = 0.0
    one = 1.0
    two = 2.0
    half = 0.5
    
    return

#-----------------------------------------------------------------------------#
# Riemann solver: Roe's approximate Riemann solver
# this is independent of Katate Masatsuka's Roe solver
#-----------------------------------------------------------------------------#
def roe1D(nx,gamma,uL,uR,f,fL,fR) :
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