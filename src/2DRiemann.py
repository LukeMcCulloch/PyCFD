#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 11:15:58 2019

@author: lukemcculloch
"""

# NORMAL AND TRANSVERSE RIEMANN SOLVER SOLUTION (cartesian grid)
#import numpy
import numpy as np

# Set left and right state and deltaq (q = [sig11,sig22,sig12,u,v])
qL = np.array([1.0, 0.0, 0.0])
qR = np.array([0.0, 0.0, 0.0])
dq = np.array([qR[0]-qL[0], 
               qR[1]-qL[1],
               qR[2]-qL[2]])

# Set normal for normal solver
nx = 1.0
ny = 0.0
# Set normal for transverse solver
nxt = 0.0
nyt = 1.0

# Set grid size (dx2=dx/2) and time step
dx2 = 0.01
dy2 = 0.01
dt = 0.01

# Set bulk and density (left and right)
bulkL = 1;   rhoL = 1 
bulkR = 8;   rhoR = 2

# Define speeds and impedance (left and right)
cL = np.sqrt(bulkL/rhoL);  ZL = rhoL*cL
cR = np.sqrt(bulkR/rhoR);  ZR = rhoR*cR

## NORMAL SOLVER

# Define the 2 eigenvectors (from columns of Matrix R)
rL = np.array([-ZL, nx, ny])
rR = np.array([ZR, nx, ny])
# Define eigenvalues
sL = -cL
sR = cR

# Compute the 2 alphas
det = ZL + ZR
alL = (-dq[0] + (nx*dq[1] + ny*dq[2])*ZR)/det
alR = (dq[0] +  (nx*dq[1] + ny*dq[2])*ZL)/det

# Compute wave fluctuations
amdq = alL*rL*sL
apdq = alR*rR*sR

## TRANSVERSE SOLVER (assuming same material on top and bottom cells)
# Define the 2 eigenvectors (from columns of Matrix R)
rB = np.array([-ZR, nxt, nyt])
rU = np.array([ZR, nxt, nyt])
# Define eigenvalues
sB = -cR
sU = cR

det = 2.0*ZR
beB = (-apdq[0] + (nxt*apdq[1] + nyt*apdq[2])*ZR)/det
beU = (apdq[0] +  (nxt*apdq[1] + nyt*apdq[2])*ZR)/det

# Compute transverse wave fluctuations
bmdq = beB*rB*sB
bpdq = beU*rU*sU