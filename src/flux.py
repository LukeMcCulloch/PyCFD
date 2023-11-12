#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:15:20 2020

"""
import numpy as np
Array = np.zeros
sqrt = np.sqrt

from Utilities import u2w



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
def roe3D(ucL, ucR, njk, num_flux, wsn, gamma = 1.4):
    """
    3D Roe approximate Riemann Solver for 
    the flux across a face
    
    input:
    ucL = np.zeros(5,float)         #conservative variables in 3D
    ucR = np.zeros(5,float)         #conservative variables in 3D
    njk = np.zeros(3,float)         #face normal in 3D
    
    output:
    num_flux array(4,float)         #Numerical viscous flux
    wsn                             # max wave speed
    
    
    #---------------
    # debugging
    sqrt = np.sqrt
    
    ucL = self.uL3d
    ucR = self.uR3d
    njk = self.n12_3d
    num_flux = self.num_flux3d
    wsn = self.wsn
    gamma = 1.4
    
    
    
    
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
    fL                      = np.zeros(5,float)     # Physical flux evaluated at ucL
    fR                      = np.zeros(5,float)     # Physical flux evaluated at ucR
    
    RT                      = 0.                    # RT = sqrt(rhoR/rhoL)
    rho,u,v,w,H,a,qn        = 0.,0.,0.,0.,0.,0.,0.  # Roe-averages
    
    rho, dqn, dp            = 0.,0.,0.              # Differences in rho, qn, p, e.g., dp=pR-pL
    du, dv, dw              = 0.,0.,0.              # Velocity differences
    
    LdU     = np.zeros(4,float)     # Wave strengths = L*(UR-UL)
    ws      = np.zeros(4,float)     # Wave speeds
    dws     = np.zeros(4,float)     # Width of a parabolic fit for entropy fix 
    R       = np.zeros((5,4),float) # Right-eigenvector matrix 
    diss    = np.zeros(5,float)     # Dissipation term
    
    #eigen limiter
    #eig_limiting_factor = np.asarray([ 0.1, 0.1, 0.1, 0.1, 0.1 ]) #eigenvalue limiting factor
    eig_limiting_factor = np.asarray([ 0.2, 0.2, 0.2, 0.2, 0.2 ]) #TLM todo: get from inputs iml!!
    
    # Face normal vector (unit vector)
    #nx,ny,nz = njk
    nx = njk[0]
    ny = njk[1]
    nz = njk[2]
    
    
    #Primitive and other variables.
    
    assert(ucL[0] != 0.0),"ERROR: ucL[0] :: rho=0.0 :: / by zero issue"
    assert(ucR[0] != 0.0),"ERROR: ucR[0] :: rho=0.0 :: / by zero issue"
    
    # if ucL[0] == 0.0: 
    #     ucL[0] = 1.e-15
    #     print('setting Left density to 1e-15 to fix devide by zero in roe flux')
    # if ucR[0] == 0.0: 
    #     ucR[0] = 1.e-15
    #     print('setting Right density to 1e-15 to fix devide by zero in roe flux')
    
    #print('ucL',ucL)
    #  Left state
    rhoL = ucL[0]
    uL   = ucL[1]/ucL[0]
    vL   = ucL[2]/ucL[0]
    wL   = ucL[3]/ucL[0]
    qnL = uL*nx + vL*ny + wL*nz
    ##print('one, ucL[4], half, rhoL, uL,uL,vL,vL,wL,wL')
    ##print(one, ucL[4], half, rhoL, uL,uL,vL,vL,wL,wL)
    #print('uL*uL+vL*vL+wL*wL',uL*uL+vL*vL+wL*wL)
    #print('half*rhoL*(uL*uL+vL*vL+wL*wL)',half*rhoL*(uL*uL+vL*vL+wL*wL))
    pL = (gamma-one)*( ucL[4] - half*rhoL*(uL*uL+vL*vL+wL*wL) )
    #print('pL',pL)
    #print('gamma*pL/rhoL',gamma*pL/rhoL)
    #assert(gamma*pL/rhoL>0.0),"gamma = {}, pL = {}, rhoL = {}".format(gamma, pL, rhoL)
    if (gamma*pL/rhoL<0.0):
        #print("ERROR: gamma = {}, pL = {}, rhoL = {}".format(gamma, pL, rhoL))
        aL = sqrt(abs(gamma*pL/rhoL))
    else:
        aL = sqrt(gamma*pL/rhoL)
    ##pL = abs((gamma-one)*( ucL[4] - half*rhoL*(uL*uL+vL*vL+wL*wL) ))
    ##aL = sqrt(abs(gamma*pL/rhoL) )
    HL = aL*aL/(gamma-one) + half*(uL*uL+vL*vL+wL*wL)
    
    
    #print('ucR',ucR)
    #  Right state
    rhoR = ucR[0]
    uR   = ucR[1]/ucR[0]
    vR   = ucR[2]/ucR[0]
    wR   = ucR[3]/ucR[0]
    qnR = uR*nx + vR*ny + wR*nz
    ##print('one, ucL[4], half, rhoR, uR,uR,vR,vR,wR,wR')
    ##print(one, ucL[4], half, rhoR, uR,uR,vR,vR,wR,wR)
    #print('uR*uR+vR*vR+wR*wR',uR*uR+vR*vR+wR*wR)
    #print('half*rhoR*(uR*uR+vR*vR+wR*wR)',half*rhoR*(uR*uR+vR*vR+wR*wR))
    pR = (gamma-one)*( ucR[4] - half*rhoR*(uR*uR+vR*vR+wR*wR) )
    #print('pR',pR)
    #print('gamma*pR/rhoR',gamma*pR/rhoR)
    assert(gamma*pR/rhoL>0.0),"gamma = {}, pR = {}, rhoR = {}".format(gamma, pR, rhoR)
    if (gamma*pR/rhoL<0.0):
        print("ERROR: gamma = {}, pR = {}, rhoR = {}".format(gamma, pR, rhoR))
        aR = sqrt(abs(gamma*pR/rhoR))
    else:       
        aR = sqrt(gamma*pR/rhoR)
    ##pR = abs( (gamma-one)*( ucR[4] - half*rhoR*(uR*uR+vR*vR+wR*wR) ) )
    ##aR = sqrt(abs(gamma*pR/rhoR))
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
    #RT = sqrt(abs(rhoR/rhoL) )
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
    #print('a = ',a)
    LdU[0] = (dp - rho*a*dqn )/(two*a*a) #Left-moving acoustic wave strength
    #print('LdU[0] = ',LdU[0])
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
    #dws[:] = eig_limiting_factor[:-1]*a
    for i in range(4):
        dws[i] = eig_limiting_factor[i]*a
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
    
    diss[:] = ws[0]*LdU[0]*R[:,0] + ws[1]*LdU[1]*R[:,1] \
             + ws[2]*LdU[2]*R[:,2] + ws[3]*LdU[3]*R[:,3]
    
    # This is the numerical flux: Roe flux = 1/2 *[  Fn[UL]+Fn[UR] - |An|[UR-UL] ]
    
    num_flux = half * (fL + fR - diss)
    
    # Max wave speed normal to the face:
    wsn = abs(qn) + a
        
        
    return num_flux, wsn


##
#
# downsample the 3d data to 2d and pass conservative variables to primative
#
##
def roe2D(ucL3D, ucR3D, njk, num_flux, wsn, gamma = 1.4):
    
    ucL2 = np.zeros((4),float)
    ucR2 = np.zeros((4),float)
    
    
    #Left state: 2D <- 3D

    ucL2[0] = ucL3D[0]
    ucL2[1] = ucL3D[1]
    ucL2[2] = ucL3D[2]
    ucL2[3] = ucL3D[4]

    #Right state: 3D <- 2D
    
    ucR2[0] = ucR3D[0]
    ucR2[1] = ucR3D[1]
    ucR2[2] = ucR3D[2]
    ucR2[3] = ucR3D[4]
    
    
    primL = u2w(ucL2)
    primR = u2w(ucR2)
    
    # primL = np.zeros((4),float)
    # primL[0] = primL5[0]
    # primL[1] = primL5[1]
    # primL[2] = primL5[2]
    # primL[3] = primL5[4]
    
    # primR = np.zeros((4),float)
    # primR[0] = primR5[0]
    # primR[1] = primR5[1]
    # primR[2] = primR5[2]
    # primR[3] = primR5[4]
    
    njk2D = np.zeros((2),float)
    njk2D[0] = njk[0]
    njk2D[1] = njk[1]
    
    flux, wsn = roe_primative(primL, primR, njk, gamma = 1.4)
    
    # back to 3D again!
    num_flux = np.zeros((5),float)
    num_flux[0] = flux[0]
    num_flux[1] = flux[1]
    num_flux[2] = flux[2]
    num_flux[4] = flux[3] 
    
    return  num_flux, wsn
    
#********************************************************************************
#* -- Roe's Flux Function with entropy fix---
#*
#* P. L. Roe, Approximate Riemann Solvers, Parameter Vectors and Difference
#* Schemes, Journal of Computational Physics, 43, pp. 357-372.
#*
#* NOTE: 3D version of this subroutine is available for download at
#*       http://cfdbooks.com/cfdcodes.html
#*
#* ------------------------------------------------------------------------------
#*  Input:   primL(1:5) =  left state (rhoL, uL, vL, pL)
#*           primR(1:5) = right state (rhoR, uR, vR, pR)
#*               njk(2) = Face normal (L -> R). Must be a unit vector.
#*
#* Output:    flux(1:5) = numerical flux
#*                  wsn = half the max wave speed
#*                        (to be used for time step calculations)
#* ------------------------------------------------------------------------------
#*
#********************************************************************************
def roe_primative(primL, primR, njk, gamma = 1.4):
    '''
    primative variable formulation of the Roe flux: 
        approximate Riemann solver for flux across a face

    Parameters
    ----------
    
    primL : primative variable left state 
    primR : primative variable right state 
    
    
    njk : normal vector pointing out of the face
    flux : delta w
    wsn : wave speed
    gamma : air is 1.4.

    Returns
    -------
    flux : delta
    wsn : wave speed

    '''
    zero = 0.0
    one = 1.0
    half = 0.5
    two = 2.0
    #fourth = 1./4.
    fifth = 1./5.
    
    LdU = np.zeros((4),float)   #wave strengths
    diss = np.zeros((4),float)  # Fluxes ad dissipation term
    ws = np.zeros((4),float)    # Wave speeds
    Rv = np.zeros((4,4),float)  # right-eigevectors
    fL = np.zeros((4),float)    # Fluxes ad dissipation term
    fR = np.zeros((4),float)    # Fluxes ad dissipation term
    dws = np.zeros((4),float)   # User-specified width for entropy fix
    
    nx = njk[0]
    ny = njk[1]
    
    # Tangent vector (Do you like it? Actually, Roe flux can be implemented 
    #   without any tangent vector. See "I do like CFD, VOL.1" for details.)
    mx = -ny
    my =  nx
    
    
    #Primitive and other variables.
    #  Left state
    rhoL = primL[0]
    uL = primL[1]
    vL = primL[2]
    unL = uL*nx+vL*ny
    umL = uL*mx+vL*my
    pL = primL[3]
    aL = sqrt(gamma*pL/rhoL)
    HL = aL*aL/(gamma-one) + half*(uL*uL+vL*vL)
    #  Right state
    rhoR = primR[0]
    uR = primR[1]
    vR = primR[2]
    unR = uR*nx+vR*ny
    umR = uR*mx+vR*my
    pR = primR[3]
    aR = sqrt(gamma*pR/rhoR)
    HR = aR*aR/(gamma-one) + half*(uR*uR+vR*vR)
    
    # compute the Roe Averages
    RT = sqrt(rhoR/rhoL)
    rho = RT*rhoL
    u = (uL+RT*uR)/(one+RT)
    v = (vL+RT*vR)/(one+RT)
    H = (HL+RT* HR)/(one+RT)
    a = sqrt( (gamma-one)*(H-half*(u*u+v*v)) )
    un = u*nx+v*ny
    um = u*mx+v*my
    
    #Wave Strengths
    drho = rhoR - rhoL 
    dp =   pR - pL
    dun =  unR - unL
    dum =  umR - umL
    #print('primative a = ',a)
    LdU[0] = (dp - rho*a*dun )/(two*a*a)
    #print('primative LdU[0] = ',LdU[0])
    LdU[1] = rho*dum
    LdU[2] =  drho - dp/(a*a)
    LdU[3] = (dp + rho*a*dun )/(two*a*a)
    
    #Wave Speed
    ws[0] = abs(un-a)
    ws[1] = abs(un)
    ws[2] = abs(un)
    ws[3] = abs(un+a)
    
    #Harten's Entropy Fix JCP(1983), 49, pp357-393:
    # only for the nonlinear fields.
    dws[0] = fifth
    if ( ws[0] < dws[0] ): ws[0] = half * ( ws[0]*ws[0]/dws[0]+dws[0] )
    dws[3] = fifth
    if ( ws[3] < dws[3] ): ws[3] = half * ( ws[3]*ws[3]/dws[3]+dws[3] )
    
    #Right Eigenvectors
    Rv[0,0] = one    
    Rv[1,0] = u - a*nx
    Rv[2,0] = v - a*ny
    Rv[3,0] = H - un*a
    
    Rv[0,1] = zero
    Rv[1,1] = mx
    Rv[2,1] = my
    Rv[3,1] = um
    
    Rv[0,2] = one
    Rv[1,2] = u
    Rv[2,2] = v 
    Rv[3,2] = half*(u*u+v*v)
    
    Rv[0,3] = one
    Rv[1,3] = u + a*nx
    Rv[2,3] = v + a*ny
    Rv[3,3] = H + un*a
    
    #Dissipation Term
    diss[:] = 0.0
    for i in range(4):
        for j in range(4):
            diss[i] += ws[j]*LdU[j]*Rv[i,j]
            
    #Compute the flux.
    fL[0] = rhoL*unL
    fL[1] = rhoL*unL * uL + pL*nx
    fL[2] = rhoL*unL * vL + pL*ny
    fL[3] = rhoL*unL * HL
    
    fR[0] = rhoR*unR
    fR[1] = rhoR*unR * uR + pR*nx
    fR[2] = rhoR*unR * vR + pR*ny
    fR[3] = rhoR*unR * HR
    
    flux = half * (fL[:] + fR[:] - diss[:])
    wsn = half*(abs(un) + a)  #Normal max wave speed times half
    return flux, wsn
    
    




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