# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:21:51 2023

@author: LUKE.MCCULLOCH
"""

import numpy as np

def compute_manufactured_sol_and_f_euler(x,y,f):
    
    pi = np.pi
    half = 0.5
    one = 1.0
    
    w = np.zeros(4)
    #f = np.zeros(4)
    
    
    #-----------------------------------------------------------
    # Ratio of specific heats for air. Let's assume air.
    
    gamma = 1.4
    
    #-----------------------------------------------------------
    # Define some indices

    iconti =  0 # continuity equation
    ixmom  =  1 # x-momentum equation
    iymom  =  2 # y-momentum equation
    ienrgy =  3 #     energy equation
    
    #-----------------------------------------------------------
    # Constants for the exact solution: c0 + cs*sin(cx*x+cy*y).
    #
    # Note: Make sure the density and pressure are positive.
    # Note: These values are passed to the subroutine:
    #         manufactured_sol(c0,cs,cx,cy, nx,ny,x,y),
    #       whcih returns the solution value or derivatives.
    
    #-----------------------------------------
    # Density    = cr0 + crs*sin(crx*x+cry*y)
    
    cr0 =  1.12
    crs =  0.15
    crx =  3.12*pi
    cry =  2.92*pi
    
    #-----------------------------------------
    # X-velocity = cu0 + cus*sin(cux*x+cuy*y)
       
    cu0 =  1.32
    cus =  0.06
    cux =  2.09*pi
    cuy =  3.12*pi
       
    #-----------------------------------------
    # Y-velocity = cv0 + cvs*sin(cvx*x+cvy*y)
       
    cv0 =  1.18
    cvs =  0.03
    cvx =  2.15*pi
    cvy =  3.32*pi
       
    #-----------------------------------------
    # Pressure   = cp0 + cps*sin(cpx*x+cpy*y)
       
    cp0 =  1.62
    cps =  0.31
    cpx =  3.79*pi
    cpy =  2.98*pi
    
    
        
        
    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    # Part I: Compute w = [rho,u,v,p] and grad(w).
    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    
    #------------------------------------------------------------------------
    # rho: Density and its derivatives
    
    rho = manufactured_sol(cr0,crs,crx,cry, 0,0,x,y)
    rhox = manufactured_sol(cr0,crs,crx,cry, 1,0,x,y)
    rhoy = manufactured_sol(cr0,crs,crx,cry, 0,1,x,y)
    
    
    #------------------------------------------------------------------------
    # u: x-velocity and its derivatives
    
    u  = manufactured_sol(cu0,cus,cux,cuy, 0,0,x,y)
    ux = manufactured_sol(cu0,cus,cux,cuy, 1,0,x,y)
    uy = manufactured_sol(cu0,cus,cux,cuy, 0,1,x,y)
    
    #------------------------------------------------------------------------
    # v: y-velocity and its derivatives
    
    v  = manufactured_sol(cv0,cvs,cvx,cvy, 0,0,x,y)
    vx = manufactured_sol(cv0,cvs,cvx,cvy, 1,0,x,y)
    vy = manufactured_sol(cv0,cvs,cvx,cvy, 0,1,x,y)
    
    #------------------------------------------------------------------------
    # p: pressure and its derivatives
    
    p  = manufactured_sol(cp0,cps,cpx,cpy, 0,0,x,y)
    px = manufactured_sol(cp0,cps,cpx,cpy, 1,0,x,y)
    py = manufactured_sol(cp0,cps,cpx,cpy, 0,1,x,y)
    
    #------------------------------------------------------------------------
    # Store the exact solution in the array for return.
    
    w = np.asarray([ rho, u, v, p ])
    
    
    print('rho, rhox, rhoy = ', rho, rhox, rhoy)
    print('u, ux, uy = ', u, ux, uy)
    print('u, ux, uy = ', u, ux, uy)
    print('p, px, py = ', p, px, py)
 
    #-----------------------------------------------------------------------------
    #
    # Exact (manufactured) solutons and derivatives have been computed.
    # We move onto the forcing terms, which are the governing equations we wish
    # to solve (the Euler) evaluated by the exact solution (i.e., 
    # the function we wish to make exact with the forcing terms). See below.
    #
    # Euler:  dF(w)/dx + dG(w)/dy = 0.
    #
    # Our manufactured solutions, wm, are the exact soltuions to the following:
    #
    #   dF(w)/dx + dG(w)/dy = f,
    #
    # where f = dF(wm)/dx + dG(wm)/dy.
    #
    # So, if we solve
    #
    #   dF(w)/dx + dG(w)/dy = f
    #
    # then we can measure the discretization error (solution error): |w-wm|.
    #
    #-----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    # Part II: Compute the forcing terms.
    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    #  Inviscid terms
    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    
    # The subroutine 'derivatives_ab" computes the product and derivatives of
    # two variables (a,b): a*b, ax*b+a*bx, ay*b+a*by.
    

    # Derivatives of u^2
    u2,u2x,u2y = derivatives_ab(u,ux,uy, u,ux,uy)

    # Derivatives of v^2
    v2,v2x,v2y = derivatives_ab(v,vx,vy, v,vx,vy)

    # Derivatives of k=(u^2+v^2)/2

    k     = half*(u*u  + v*v)
    kx    = half*(u2x   + v2x)
    ky    = half*(u2y   + v2y)

    # Derivatives of rho*k = rho*(u^2+v^2)/2
    a,ax,ay = derivatives_ab(rho,rhox,rhoy, k,kx,ky) #a=rho*(u^2+v^2)/2

    # Derivatives of rho*H = gamma/(gamma-1)*p + rho*k
    rhoH    = gamma/(gamma-one)*p    + a
    rhoHx   = gamma/(gamma-one)*px   + ax
    rhoHy   = gamma/(gamma-one)*py   + ay

    #-----------------------------------------------------------------------------

    # Compute derivatives of (rho*u)
    a,ax,ay = derivatives_ab(rho,rhox,rhoy, u,ux,uy) #a=(rho*u)

    # Compute derivatives of (rho*v)
    b,bx,by = derivatives_ab(rho,rhox,rhoy, v,vx,vy) #b=(rho*v)

    #-----------------------------------------------------------------------------

    # Compute derivatives of (a*u)=(rho*u*u) #a=(rho*u)

    au,aux,auy = derivatives_ab(a,ax,ay, u,ux,uy)

    # Compute derivatives of (a*v)=(rho*u*v) #a=(rho*u)

    av,avx,avy = derivatives_ab(a,ax,ay, v,vx,vy)

    # Compute derivatives of (b*u)=(rho*v*u) #b=(rho*v)

    bu,bux,buy = derivatives_ab(b,bx,by,  u,ux,uy)

    # Compute derivatives of (b*v)=(rho*v*v) #b=(rho*v)

    bv,bvx,bvy = derivatives_ab(b,bx, by, v,vx,vy)

    #-----------------------------------------------------------------------------

    # Compute derivatives of (u*rH)
    
    rhouH,rhouHx,rhouHy = derivatives_ab( u,ux,uy, rhoH,rhoHx,rhoHy)
    
    # Compute derivatives of (v*rH)
    
    rhovH,rhovHx,rhovHy = derivatives_ab( v,vx,vy, rhoH,rhoHx,rhoHy)
    
    #---------------------------------------------------------------------
    
    #---------------------------------------------------------------------
    # Store the inviscid terms in the forcing term array, f(:).
    #---------------------------------------------------------------------
    
    #------------------------------------------------------
    # Continuity:         (rho*u)_x   +   (rho*v)_y
    f[iconti]  = (rhox*u + rho*ux) + (rhoy*v + rho*vy)
    
    #------------------------------------------------------
    # Momentum:     (rho*u*u)_x + (rho*u*v)_y + px
    f[ixmom]   =     aux     +    buy      + px
    
    #------------------------------------------------------
    # Momentum:     (rho*u*v)_x + (rho*v*v)_y + px
    f[iymom]   =     avx     +    bvy      + py
    
    #------------------------------------------------------
    # Energy:       (rho*u*H)_x + (rho*v*H)
    f[ienrgy]  =    rhouHx   +   rhovHy
    
    
    return w[:],f[:]
       
    # Note: Later, we'll perform the following to compute the residual for
    #                   dF(w)/dx + dG(w)/dy = f
    #       Step 1. Comptue the residual: Res=dF(w)/dx + dG(w)/dy.
    #       Step 2. Subtract f: Res = Res - f.
    #

    
    
    
    

def derivatives_ab(a,ax,ay,  b,bx,by):
    '''
    #********************************************************************************
    #
    # This subroutine computes first derivatives of a quadratic term
    #
    #  Input: a, ax, ay #Function value a, and its derivatives, (ax,ay).
    #         b, bx, by #Function value b, and its derivatives, (bx,by).
    #
    # Output: ab = a*b, abx = d(a*b)/dx, aby = d(a*b)/dy.
    #
    #********************************************************************************
    '''

    
    ab    = a*b 
    abx   = ax*b + a*bx
    aby   = ay*b + a*by

    return ab,abx,aby



def manufactured_sol(a0,as_,ax,ay, nx,ny,x,y) :
    '''
    #********************************************************************************
    #* This function computes the sine function:
    #*
    #*       f =  a0 + as_*sin(ax*x+ay*y)
    #*
    #* and its derivatives:
    #*
    #*     df/dx^nx/dy^ny = d^{nx+ny}(a0+as_*sin(ax*x+ay*y))/(dx^nx*dy^ny)
    #*
    #* depending on the input parameters:
    #*
    #*
    #* Input:
    #*
    #*  a0,as_,ax,ay = coefficients in the function: f =  a0 + as_*sin(ax*x+ay*y).
    #*            x = x-coordinate at which the function/derivative is evaluated.
    #*            y = y-coordinate at which the function/derivative is evaluated.
    #*           nx = nx-th derivative with respect to x (nx >= 0).
    #*           ny = ny-th derivative with respect to y (ny >= 0).
    #*
    #* Output: The function value, fval
    #*
    #*
    #* Below are some examples:
    #*
    #*     f =  a0 + as_*sin(ax*x+ay*y)            #<- (nx,ny)=(0,0)
    #*
    #*    fx =  ax * as_*cos(ax*x+ay*y)            #<- (nx,ny)=(1,0)
    #*    fy =  ay * as_*cos(ax*x+ay*y)            #<- (nx,ny)=(0,1)
    #*
    #*   fxx = -ax**2 * as_*sin(ax*x+ay*y)         #<- (nx,ny)=(2,0)
    #*   fxy = -ax*ay * as_*sin(ax*x+ay*y)         #<- (nx,ny)=(1,1)
    #*   fyy = -ay**2 * as_*sin(ax*x+ay*y)         #<- (nx,ny)=(0,2)
    #*
    #*  fxxx = -ax**3        * as_*cos(ax*x+ay*y)  #<- (nx,ny)=(3,0)
    #*  fxxy = -ax**2 *ay    * as_*cos(ax*x+ay*y)  #<- (nx,ny)=(2,1)
    #*  fxyy = -ax    *ay**2 * as_*cos(ax*x+ay*y)  #<- (nx,ny)=(1,2)
    #*  fyyy = -       ay**3 * as_*cos(ax*x+ay*y)  #<- (nx,ny)=(0,3)
    #*
    #* fxxxx =  ax**4        * as_*sin(ax*x+ay*y)  #<- (nx,ny)=(4,0)
    #* fxxxy =  ax**3 *ay    * as_*sin(ax*x+ay*y)  #<- (nx,ny)=(3,1)
    #* fxxyy =  ax**2 *ay**2 * as_*sin(ax*x+ay*y)  #<- (nx,ny)=(2,2)
    #* fxyyy =  ax    *ay**3 * as_*sin(ax*x+ay*y)  #<- (nx,ny)=(1,3)
    #* fyyyy =         ay**4 * as_*sin(ax*x+ay*y)  #<- (nx,ny)=(0,4)
    #*
    #* and so on.
    #*
    #*
    #********************************************************************************
    '''
    
    if (nx < 0 or ny < 0) :
        print(" Invalid input: nx and ny must be greater or equal to zero... Try again.")
        assert(0)
    
    if ( nx+ny == 0 ) :
    
        fval = a0 + as_*np.sin(ax*x + ay*y)
    
    elif ( np.mod(nx+ny,2) == 0 ) :
    
        fval = - (ax**nx * ay**ny)*as_*np.sin(ax*x + ay*y)
        if ( np.mod(nx+ny,4)   == 0 ) : fval = -fval
    
    else:
       
        fval = (ax**nx * ay**ny)*as_*np.cos(ax*x + ay*y)
        if ( np.mod(nx+ny+1,4) == 0 ) : fval = -fval
    
    
    return fval
#********************************************************************************

