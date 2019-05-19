#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 07:21:28 2019

@author: lukemcculloch
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
zeros = np.zeros

def max2d(this):
    return max(map(max, this))
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
nstep=100
maxit=200
maxError=0.001
beta=1.2
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
for i in range(0,nx+2):
    x[i]=dx*(i-1.5) 
for j in range(0,ny+2):
    y[j]=dy*(j-1.5)

#Set density
r=zeros((nx+2,ny+2),float)+rho1
for i in range(1,nx+1):
    for j in range(1,ny+1):
        if ( (x[i]-xc)**2+(y[j]-yc)**2 < rad**2):
            r[i,j]=rho2
#================== START TIME LOOP======================================
for iis in range(1,nstep):#is
    print iis
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
    rt[:nx+1,0]=lrg
    rt[:nx+1,ny+1]=lrg
    rt[1,1:ny+2]=lrg
    rt[nx+1,:ny+1]=lrg
    for i in range(1,nx+1):
        for j in range(1,ny+1):
            tmp1[i,j]= (0.5/dt)*( (ut[i,j]-ut[i-1,j])/dx+(vt[i,j]-vt[i,j-1])/dy )
            tmp2[i,j]=1.0/( (1./dx)*( 1./(dx*(rt[i+1,j]+rt[i,j]))+
                            1./(dx*(rt[i-1,j]+rt[i,j]))  )+
                        (1./dy)*(1./(dy*(rt[i,j+1]+rt[i,j]))+
                        1./(dy*(rt[i,j-1]+rt[i,j]))   )   )
    
    for it in range(1,maxit):                # SOLVE FOR PRESSURE
        oldArray=p;
        for i in range(1,nx+1):
            for j in range(1,ny+1):
                p[i,j]=(1.0-beta)*p[i,j]+beta* tmp2[i,j]*(
                  (1./dx)*( p[i+1,j]/(dx*(rt[i+1,j]+rt[i,j]))+
                  p[i-1,j]/(dx*(rt[i-1,j]+rt[i,j])) )+
                  (1./dy)*( p[i,j+1]/(dy*(rt[i,j+1]+rt[i,j]))+
                  p[i,j-1]/(dy*(rt[i,j-1]+rt[i,j])) ) - tmp1[i,j])
    
        if np.any( max(map(max, oldArray-p)) <maxError ): break
   
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