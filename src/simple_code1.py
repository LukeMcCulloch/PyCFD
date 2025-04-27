#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 07:21:28 2019

@author: lukemcculloch

this is oooold python 2.7 code.  Looks like a total 'make a solver in 1 file' type of thing

not sure if it belongs in the PyCFD codebase
I don't think it's useful at this point (April 2025)
"""
#========================================================================#
#code1.m
# A very simple Navier-Stokes solver for a drop falling in a rectangular
# domain. 
#The viscosity is taken to be a constant and a forward in time,
# centered in space discretization is used. The density is advected by a
# simple upwind scheme.
#========================================================================
#
import numpy as np
from pylab import * #old fashioned quiver example
zeros = np.zeros

#from AdaptiveMeshRefinement import AMR, cell_reconstruct_gradient


def max2d(this):
    return max(map(max, abs(this)))
#domain size and physical variables
Lx=1.0
Ly=1.0
gx=0.0
gy=-100.0 
rho1=1.0
rho2=2.0
m0=0.01
rro=rho1
unorth=0.
usouth=0.
veast=0.
vwest=0.
#
#Numerical variables
nx=32
ny=32
dt=0.00125
nstep=20
maxit=200
maxError=0.001
beta=1.2
#
stepx = Lx/nx
stepy = Ly/ny
#
# Initial drop size and location
time=0.0
rad=0.15
xc=0.5
yc=0.7
#
# Zero various arrys
x = zeros((nx+2),float)
y = zeros((ny+2),float)
u=zeros((nx+1,ny+2),float)
v=zeros((nx+2,ny+1),float)
p=zeros((nx+2,ny+2),float)
ut=zeros((nx+1,ny+2),float)
vt=zeros((nx+2,ny+1),float)
tmp1=zeros((nx+2,ny+2),float)
uu=zeros((nx+1,ny+1),float)
vv=zeros((nx+1,ny+1),float)
tmp2=zeros((nx+2,ny+2),float)
# Zero various arrys
#
# Set the gridd
dx=Lx/nx
dy=Ly/ny
for i in range(nx+2):
    x[i]=dx*(i-1.5) 
for j in range(ny+2):
    y[j]=dy*(j-1.5)

#Set density
r=zeros((nx+2,ny+2),float)+rho1
for i in range(1,nx+1):
    for j in range(1,ny+1):
        if ( (x[i]-xc)**2+(y[j]-yc)**2 < rad**2):
            r[i,j]=rho2
#================== START TIME LOOP======================================
for iis in range(nstep):#is
    print( iis )
    # tangential velocity at boundaries
    u[:nx+1,1]=2*usouth-u[:nx+1,2]
    u[:nx+1,ny+1]=2*unorth-u[:nx+1,ny+1]
    v[0,:ny+1]=2*vwest-v[1,0:ny+1]
    v[nx+1,:ny+1]=2*veast-v[nx+1,:ny+1]
  
    for i in range(1,nx): #tlm
        for j in range(1,ny+1):      # TEMPORARY u-velocity
            ut[i,j]=u[i,j]+dt*(-0.25*(((u[i+1,j]+u[i,j])**2-(u[i,j]+   
            u[i-1,j])**2)/dx+((u[i,j+1]+u[i,j])*(v[i+1,j]+         
             v[i,j])-(u[i,j]+u[i,j-1])*(v[i+1,j-1]+v[i,j-1]))/dy)+ 
                m0/(0.5*(r[i+1,j]+r[i,j]))*(                          
                        (u[i+1,j]-2*u[i,j]+u[i-1,j])/dx**2+            
                        (u[i,j+1]-2*u[i,j]+u[i,j-1])/dy**2 )+gx    )
    
    for i in range(1,nx+1): #TLM
        for j in range(1,ny):       # TEMPORARY v-velocity
            vt[i,j]=v[i,j]+dt*(-0.25*(((u[i,j+1]+u[i,j])*(v[i+1,j]+   
              v[i,j])-(u[i-1,j+1]+u[i-1,j])*(v[i,j]+v[i-1,j]))/dx+  
            ((v[i,j+1]+v[i,j])**2-(v[i,j]+v[i,j-1])**2)/dy)+      
            m0/(0.5*(r[i,j+1]+r[i,j]))*(                          
                    (v[i+1,j]-2*v[i,j]+v[i-1,j])/dx**2+            
                    (v[i,j+1]-2*v[i,j]+v[i,j-1])/dy**2 )+gy    )
    #========================================================================
    # Compute source term and the coefficient for p[i,j]
    rt=r
    lrg=1000
    rt[:nx+2,0]=lrg
    rt[:nx+2,ny+1]=lrg #use -1 for clarity
    rt[0,:ny+2]=lrg
    rt[nx+1,:ny+2]=lrg
    for i in range(1,nx+1):
        for j in range(1,ny+1):
            tmp1[i,j]= (0.5/dt)*( (ut[i,j]-ut[i-1,j])/dx+(vt[i,j]-vt[i,j-1])/dy )
            tmp2[i,j]=1.0/( (1./dx)*( 1./(dx*(rt[i+1,j]+rt[i,j]))+
                            1./(dx*(rt[i-1,j]+rt[i,j]))  )+
                        (1./dy)*(1./(dy*(rt[i,j+1]+rt[i,j]))+
                        1./(dy*(rt[i,j-1]+rt[i,j]))   )   )
    
    for it in range(maxit):                # SOLVE FOR PRESSURE
        oldArray=p;
        for i in range(1,nx+1):
            for j in range(1,ny+1):
                p[i,j]=(1.0-beta)*p[i,j] + \
                        beta* tmp2[i,j]*(
                          (1./dx)*( p[i+1,j]/(dx*(rt[i+1,j]+rt[i,j]))+
                          p[i-1,j]/(dx*(rt[i-1,j]+rt[i,j])) )+
                          (1./dy)*( p[i,j+1]/(dy*(rt[i,j+1]+rt[i,j]))+
                          p[i,j-1]/(dy*(rt[i,j-1]+rt[i,j])) ) - tmp1[i,j]
                         )
    
        if ( max2d(oldArray-p) <maxError ): break
   
    for i in range(1,nx):
        for j in range(1,ny+1):   # CORRECT THE u-velocity
            u[i,j]=ut[i,j]-dt*(2.0/dx)*(p[i+1,j]-p[i,j])/(r[i+1,j]+r[i,j]);
   
    for i in range(1,nx+1):
        for j in range(1,ny):   # CORRECT THE v-velocity
            v[i,j]=vt[i,j]-dt*(2.0/dy)*(p[i,j+1]-p[i,j])/(r[i,j+1]+r[i,j])
    
#=======ADVECT DENSITY using centered difference plus diffusion ==========
    ro=r
    for i in range(1,nx+1):
        for j in range(1,ny+1):
            r[i,j]=ro[i,j]-(0.5*dt/dx)*(u[i,j]*(ro[i+1,j]
            +ro[i,j])-u[i-1,j]*(ro[i-1,j]+ro[i,j]) )
            -(0.5* dt/dy)*(v[i,j]*(ro[i,j+1]
            +ro[i,j])-v[i,j-1]*(ro[i,j-1]+ro[i,j])  )
            +(m0*dt/dx/dx)*(ro[i+1,j]-2.0*ro[i,j]+ro[i-1,j])
            +(m0*dt/dy/dy)*(ro[i,j+1]-2.0*ro[i,j]+ro[i,j-1])

#========================================================================
time=time+dt                   # plot the results
uu[:nx+1,:ny+1]=0.5*(u[:nx+1,1:ny+2]+u[:nx+1,:ny+1])
vv[:nx+1,:ny+1]=0.5*(v[1:nx+2,:ny+1]+v[:nx+1,:ny+1])
xh = zeros((nx+1),float)
yh = zeros((ny+1),float)
for i in range(1,nx+1):
    xh[i]=dx*(i-1)
for j in range(1,ny+1):
    yh[j]=dy*(j-1)
#hold off,contour[x,y,flipud(rot90(r))),axis equal,axis([0 Lx 0 Ly]);
#hold on;quiver(xh,yh,flipud(rot90(uu)),flipud(rot90(vv)),’r’);
#pause(0.01)
    
    
# http://matplotlib.org/examples/pylab_examples/quiver_demo.html
def plotsol(x,y,u,v,xh,yh):
    startx = x[0]
    stopx = x[-1]
    starty = y[0]
    stopy = y[-1]
    X,Y = meshgrid( arange(startx,stopx,stepx),
                    arange(starty,stopy,stepy) )
    U=u
    V=v
    M=np.sqrt(pow(U[:,1:], 2) + pow(V[1:,:], 2))
    
    figure()
    if len(X)==11:
        Q = quiver( X,Y, U, V, M, units='x', pivot='tip',width=.005, scale=1./.15)
        if BC==True:
            qk = quiverkey(Q, 0.5, .94, 1, r'$u = 1 \frac{m}{s}$', labelpos='W',
                   fontproperties={'weight': 'bold'})
            qk = quiverkey(Q, 0.75, .94, 0, r'$v=0\frac{m}{sec}$ ', labelpos='W',color='w',
                   fontproperties={'weight': 'bold'})
            #qk = quiverkey(Q, .08, .5, 0, r'  $0 \frac{m}{s}$', labelpos='W',
            #       fontproperties={'weight': 'bold'})
            
            qk = quiverkey(Q, .15, .5, 0, r'  $u=v=0\frac{m}{sec}$', labelpos='W',color='w',
                   fontproperties={'weight': 'bold'})
            qk = quiverkey(Q, 1.01, .5, 0, r'  $u=v=0\frac{m}{sec}$', labelpos='W',color='w',
                   fontproperties={'weight': 'bold'})
            qk = quiverkey(Q, 0.6, .1, 0, r'  $u=v=0\frac{m}{sec}$', labelpos='W',color='w',
                   fontproperties={'weight': 'bold'})
    elif len(X)<=51:
        Q = quiver( X,Y, U, V, M, units='x', 
                   pivot='tip',width=.005, scale=2./.15)
    elif len(X)<=101:
        Q = quiver( X,Y, U, V, M, units='x', 
                   pivot='tip',width=.005, scale=3./.15)
    else:
        Q = quiver( X,Y, U, V, M, units='x', 
                   pivot='tip',width=.005, scale=3.3/.15)
          
    
    
    l,r,b,t = axis()
    dx, dy = r-l, t-b

    #axis([-.2,1.2,-.2,1.2])
    if BC==True:
        title('U & V, Vector Magnitude Initial Conditions')
        axis([l-0.2*dx, r+0.2*dx, b-0.2*dy, t+0.2*dy])
    else:
        title('U & V, Vector Magnitudes')
        axis([l-0.05*dx, r+0.05*dx, b-0.05*dy, t+0.05*dy])
    CB = plt.colorbar(Q, shrink=0.8, extend='both')
    xlabel('X location')
    ylabel('Y location')
    plt.plot(bx,by)
    plt.axis('equal')
    show()
    return

def contour(x,y,r):
    
    # Bounding Box:
    bx=[0.,Lx,Lx,0.,0.]
    by=[0.,0.,Ly,Ly,0.]
    
    startx = x[0]
    stopx = x[-1]
    starty = y[0]
    stopy = y[-1]
    streamtlm = r
    X,Y = meshgrid( arange(startx,stopx,stepx),
                    arange(starty,stopy,stepy) )
    #levels=np.linspace(amin(streamtlm),amax(streamtlm),50,endpoint=True)
    levels=np.linspace(-.105,0.,50,endpoint=True)
    #c = plt.contour(X, Y,streamtlm,10)# good for Re 10
    c = plt.contour(X, Y,streamtlm,levels)
    plt.clabel(c, inline=1, fontsize=5)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    mytitle = r"Stream Function"
    plt.title(mytitle)
    # we switch on a grid in the figure for orientation
    plt.grid()
    # colorbar
    CB = plt.colorbar(c, shrink=0.8, extend='both')
    plt.axis('equal')
    plt.plot(bx,by)
    plt.show()
    return


plotsol(xh,yh,u,v,xh,yh)