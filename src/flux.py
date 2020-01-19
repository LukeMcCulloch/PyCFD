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
    
    #Local variables
    #            L = Left
    #            R = Right
    # No subscript = Roe average
    
    nx, ny, nz              = 0.,0.,0.              # Normal vector components
    uL, uR, vL, vR, wL, wR  = 0.,0.,0.,0.,0.,0      # Velocity components.
    rhoL, rhoR, pL, pR      = 0.,0.,0.,0.           # Pimitive variables.
    qnL, qnR                = 0.,0.                 # Normal velocities
    aL, aR, HL, HR          = 0.,0.,0.,0.           # Speed of sound, Total enthalpy
    fL                      = 0.                    # Physical flux evaluated at ucL
    fR                      = 0.                    # Physical flux evaluated at ucR
    
    RT                      = 0.                    # RT = sqrt(rhoR/rhoL)
    rho,u,v,w,H,a,qn        = 0.,0.,0.,0.,0.,0.,0.  # Roe-averages
    
    rho, dqn, dp            = 0.,0.,0.              # Differences in rho, qn, p, e.g., dp=pR-pL
    du, dv, dw              = 0.,0.,0.              # Velocity differences
    
    LdU = np.zeros(4,float)     # Wave strengths = L*(UR-UL)
    ws  = np.zeros(4,float)     # Wave speeds
    dws = np.zeros(4,float)     # Width of a parabolic fit for entropy fix 
    R = np.zeros((5,4),float)   # Right-eigenvector matrix 
    diss = np.zeros(4,float)    # Dissipation term
    
    # Face normal vector (unit vector)
    nx,ny,nz = njk
    
    
    #Primitive and other variables.
    
    #  Left state
    
    rhoL = ucL[0]
    uL = ucL[1]/ucL[0]
    vL = ucL[2]/ucL[0]
    wL = ucL[3]/ucL[0]
    qnL = uL*nx + vL*ny + wL*nz
    pL = (gamma-one)*( ucL[4] - half*rhoL*(uL*uL+vL*vL+wL*wL) )
    aL = sqrt(gamma*pL/rhoL)
    HL = aL*aL/(gamma-one) + half*(uL*uL+vL*vL+wL*wL)
    
    #  Right state
    
    rhoR = ucR[0]
    uR = ucR[1]/ucR[0]
    vR = ucR[2]/ucR[0]
    wR = ucR[3]/ucR[0]
    qnR = uR*nx + vR*ny + wR*nz
    pR = (gamma-one)*( ucR[4] - half*rhoR*(uR*uR+vR*vR+wR*wR) )
    aR = sqrt(gamma*pR/rhoR)
    HR = aR*aR/(gamma-one) + half*(uR*uR+vR*vR+wR*wR)
    
    #Compute the physical flux: fL = Fn(UL) and fR = Fn(UR)
    
    fL[0] = rhoL*qnL
    fL[1] = rhoL*qnL * uL + pL*nx
    fL[2] = rhoL*qnL * vL + pL*ny
    fL[3] = rhoL*qnL * wL + pL*nz
    fL[4] = rhoL*qnL * HL
      
    fR[0] = rhoR*qnR
    fR[1] = rhoR*qnR * uR + pR*nx
    fR[2] = rhoR*qnR * vR + pR*ny
    fR[3] = rhoR*qnR * wR + pR*nz
    fR[4] = rhoR*qnR * HR
    
    #First compute the Roe-averaged quantities
    
    #  NOTE: See http://www.cfdnotes.com/cfdnotes_roe_averaged_density.html for
    #        the Roe-averaged density.
    
    
    RT = sqrt(rhoR/rhoL)
    rho = RT*rhoL                                       #Roe-averaged density
    u = (uL + RT*uR)/(one + RT)                         #Roe-averaged x-velocity
    v = (vL + RT*vR)/(one + RT)                         #Roe-averaged y-velocity
    w = (wL + RT*wR)/(one + RT)                         #Roe-averaged z-velocity
    H = (HL + RT*HR)/(one + RT)                         #Roe-averaged total enthalpy
    a = sqrt( (gamma-one)*(H-half*(u*u + v*v + w*w)) )  #Roe-averaged speed of sound
    qn = u*nx + v*ny + w*nz                             #Roe-averaged face-normal velocity
    
    #Wave Strengths
    
    drho = rhoR - rhoL #Density difference
    dp =   pR - pL   #Pressure difference
    dqn =  qnR - qnL  #Normal velocity difference
    
    LdU[0] = (dp - rho*a*dqn )/(two*a*a) #Left-moving acoustic wave strength
    LdU[1] = (dp + rho*a*dqn )/(two*a*a) #Right-moving acoustic wave strength
    LdU[2] =  drho - dp/(a*a)            #Entropy wave strength
    LdU[3] = rho                         #Shear wave strength (not really, just a factor)
    
    #Absolute values of the wave Speeds
    
    ws[0] = abs(qn-a) #Left-moving acoustic wave
    ws[1] = abs(qn+a) #Right-moving acoustic wave
    ws[2] = abs(qn)   #Entropy wave
    ws[3] = abs(qn)   #Shear waves

    # Harten's Entropy Fix JCP(1983), 49, pp357-393. This is typically applied
    # only for the nonlinear fields (k=1 and 3), but here it is applied to all
    # for robustness, avoiding vanishing wave speeds by making a parabolic fit
    # near ws = 0 for all waves.
    # 02-27-2018: The limiting can be too much for the shear wave and entropy wave.
    #             Flat plate calculation shows that applying it to all contaminates
    #             the solution significantly. So, apply only to the nonlinear waves,
    #             or apply very small limiting to entropy and shear waves.
    #
    # Note: ws(1) and ws(2) are the nonlinear waves.
    dws[:] = eig_limiting_factor[:]*a
    if ( ws[i] < dws[i] ): 
        ws[i] = half * ( ws[i]*ws[i]/dws[i]+dws[i] )
    
    #np.where( ws<dws, ws, half * ( ws[i]*ws[i]/dws[i]+dws[i] ) )
        
        
    # Right Eigenvectors
    # Note: Two shear wave components are combined into one, so that tangent vectors
    #       are not required. And that's why there are only 4 vectors here.
    #       See "I do like CFD, VOL.1" about how tangent vectors are eliminated.
        
        
    # Left-moving acoustic wave
    R[0,0] = one    
    R[1,0] = u - a*nx
    R[2,0] = v - a*ny
    R[3,0] = w - a*nz
    R[4,0] = H - a*qn

    # Right-moving acoustic wave
    R[0,1] = one
    R[1,1] = u + a*nx
    R[2,1] = v + a*ny
    R[3,1] = w + a*nz
    R[4,1] = H + a*qn

    # Entropy wave
    R[0,2] = one
    R[1,2] = u
    R[2,2] = v 
    R[3,2] = w
    R[4,2] = half*(u*u + v*v + w*w)
    
    # Two shear wave components combined into one (wave strength incorporated).
    du = uR - uL
    dv = vR - vL
    dw = wR - wL
    R[0,3] = zero
    R[1,3] = du - dqn*nx
    R[2,3] = dv - dqn*ny
    R[3,3] = dw - dqn*nz
    R[4,3] = u*du + v*dv + w*dw - qn*dqn
    
    #Dissipation Term: |An|(UR-UL) = R|Lambda|L*dU = sum_k of [ ws[k] * R[:,k] * L*dU[k] ]
    
    diss[:] = ws[0]*LdU[0]*R[:,0] + ws[1]*LdU[1]*R[:,1] &
             + ws[2]*LdU[2]*R[:,2] + ws[3]*LdU[3]*R[:,3]
    
    # This is the numerical flux: Roe flux = 1/2 *[  Fn[UL]+Fn[UR] - |An|[UR-UL] ]
    
    num_flux = half * (fL + fR - diss)
    
    # Max wave speed normal to the face:
    wsn = abs(qn) + a
        
        
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