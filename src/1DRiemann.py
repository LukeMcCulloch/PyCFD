#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 11:24:13 2019

@author: lukemcculloch
"""


# Riemman Toy IMPLEMENTATION from LaVeque
#import numpy
import numpy as np

# Set left and right state and deltaq (q = [sig11,sig22,sig12,u,v])
qL = np.array([-1.0, 0.0])
qR = np.array([1.0, 0.0])
dq = np.array([qR[0]-qL[0], qR[1]-qL[1]])

# Set bulk and density (left and right)
bulkL = 1;   rhoL = 1 
bulkR = 4;   rhoR = 2

# Define speeds and impedance (left and right)
cL = np.sqrt(bulkL/rhoL);  ZL = rhoL*cL
cR = np.sqrt(bulkR/rhoR);  ZR = rhoR*cR

# Define the 2 eigenvectors (from columns of Matrix R)
r1 = np.array([-ZL, 1])
r2 = np.array([ZR,  1])

# Compute the 2 alphas
det = ZL + ZR
alL = (-dq[0] + dq[1]*ZR)/det
alR = (dq[0] + dq[1]*ZL)/det

# Compute middle state qm
qm = qL + alL*r1 
## Should be equivalent to
#qms = qR - alR*r2 #it is!
    
# Compute waves characteristics for plotting
x = np.linspace(-5,5,50)
Wc = np.zeros((2,len(x)))
Wc[0][:] = -x/cL
Wc[1][:] = x/cR




#SOLUTION PLOTTING
# Required for plotting
#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from IPython.html.widgets import interact
#from ipywidgets import StaticInteract, RangeWidget, RadioWidget, DropDownWidget

# Plot Riemann solution
def plot_Riemann(time):
    
    fig = plt.figure(figsize=(15, 5))
    
    # Create subplot for (x,t) plane
    tmax = 5
    ax = fig.add_subplot(1,3,1)
    ax.set_xlabel('x')
    ax.set_ylabel('time')
    ax.axis([min(x),max(x),0,tmax])
    
    # Plot characteristic lines
    # Acoustic waves in red
    ax.plot(x,Wc[0][:], '-r', linewidth=2)
    ax.plot(x,Wc[1][:], '-r', linewidth=2)
    # Plot time-line in (x,t) plane
    ax.plot(x, 0*x+time, 'k', linewidth=3)

    # Create pressure subplot for Riemann solution
    ax2 = fig.add_subplot(1,3,2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Pressure')
    ax2.axis([min(x),max(x),-2,2])
    
    # Create Riemann solution vector and plot
    sol = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] < -cL*time:
            sol[i] = qL[0]
        elif x[i] < cR*time:
            sol[i] = qm[0]
        else:
            sol[i] = qR[0]
    ax2.plot(x,sol, 'k', linewidth = 3)
    ax2.fill_between(x,-20, sol, facecolor='blue', alpha=0.2)
    
    # Create velocity subplot for Riemann solution
    ax3 = fig.add_subplot(1,3,3)
    ax3.set_xlabel('x')
    ax3.set_ylabel('Velocity')
    ax3.axis([min(x),max(x),-2,2])
    
    # Create Riemann solution vector and plot
    sol = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] < -cL*time:
            sol[i] = qL[1]
        elif x[i] < cR*time:
            sol[i] = qm[1]
        else:
            sol[i] = qR[1]
    ax3.plot(x,sol, 'k', linewidth = 3)
    ax3.fill_between(x,-20, sol, facecolor='blue', alpha=0.2)
    
    return fig


if __name__ == '__main__':
    plot_Riemann(.01)
    
# Create interactive widget to visualize solution 
#interact(plot_Riemann, time=(0,5,0.1), qvar={'sigma11':0, 'sigma22':1, 'sigma12':2, 
                                             #'normal velocity u':3, 'transverse velocity v':4});
#StaticInteract(plot_Riemann, time=RangeWidget(0,5,0.25)) 