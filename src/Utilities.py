#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 15:14:18 2019

@author: luke
"""

import numpy as np

from math import pi



# An alias for np.linal.norm, because typing that is ugly
def norm(vec, *args, **kwargs):
    return np.linalg.norm(vec, *args, **kwargs)


# A quicker cross method when calling on a single vector
def cross(u, v):
    return np.array((
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0]
        ))

def dot(u,v):
    return np.dot(u,v)



'''
2D vector utilities
'''

## A wrapper for the purposes of this class, to avoid interacting with numpy
def Vector2D(x,y):
    return np.array([float(x),float(y)])

def printVec2(v):
    return "({:.5f}, {:.5f})".format(v[0], v[1])


# Normalizes a numpy vector
# This methods modifies its argument in place, but also returns a reference to that
# array for chaining.
# Works on both single vectors and nx3 arrays of vectors (perfomed in-place).
# If zeroError=False, then this function while silently return a same-sized 0
# for low-norm vectors. If zeroError=True it will throw an exception
def normalize2D(vec, zeroError=False, return_mag=False):

    # Used for testing zeroError
    eps = 0.00000000001

    # Use separate tests for 1D vs 2D arrays (TODO is there a nicer way to do this?)
    if(len(vec.shape) == 1):

        norm = np.linalg.norm(vec)
        if(norm < 0.0000001):
            if(zeroError):
                raise ArithmeticError("Cannot normalize function with norm near 0")
            else:
                vec[0] = 0
                vec[1] = 0
                return vec
        vec[0] /= norm
        vec[1] /= norm
        
        if return_mag:
            return vec, norm
        else:
            return vec

    elif(len(vec.shape) == 2):

        # Compute norms for each vector
        norms = np.sqrt( vec[:,0]**2 + vec[:,1]**2 + vec[:,2]**2 )

        # Check for norm zero, if checking is enabled
        if(zeroError and np.any(norms < 0.00000000001)):
            raise ArithmeticError("Cannot normalize function with norm near 0")

        # Normalize in place
        # oldSettings = np.seterr(invalid='ignore')    # Silence warnings since we check above if the user cares
        vec[:,0] /= norms
        vec[:,1] /= norms
        vec[:,2] /= norms
        # np.seterr(**oldSettings)

    else:
        raise ValueError("I don't know how to normalize a vector array with > 2 dimensions")

    if return_mag:
        return vec, norms
    else:
        return vec


# Normalizes a numpy vector.
# This method returns a new (normalized) vector
# Works on both single vectors and nx3 arrays of vectors (perfomed in-place).
# If zeroError=False, then this function while silently return a same-sized 0
# for low-norm vectors. If zeroError=True it will throw an exception
def normalized2D(vec, zeroError=False, return_mag=False):

    # Used for testing zeroError
    eps = 0.00000000001

    # Use separate tests for 1D vs 2D arrays (TODO is there a nicer way to do this?)
    if(len(vec.shape) == 1):

        norm = np.linalg.norm(vec)
        if(norm < 0.0000001):
            if(zeroError):
                raise ArithmeticError("Cannot normalize function with norm near 0")
            else:
                return np.zeros_like(vec)
        
        if return_mag:
            return vec / norm, norm
        else:
            return vec / norm

    elif(len(vec.shape) == 2):

        # Compute norms for each vector
        norms = np.sqrt( vec[:,0]**2 + vec[:,1]**2  )

        # Check for norm zero, if checking is enabled
        if(zeroError and np.any(norms < 0.00000000001)):
            raise ArithmeticError("Cannot normalize function with norm near 0")
        else:
            norms += 1.e-8
            #            if np.any(norms[:,0] < 0.00000000001):
            #                norms[:,0] += 1.e-8
            #            elif np.any(norms[:,1] < 0.00000000001):
            #                norms[:,1] += 1.e-8
            #            elif np.any(norms[:,2] < 0.00000000001):
            #                norms[:,2] += 1.e-8
        # Normalize in place
        # oldSettings = np.seterr(invalid='ignore')    # Silence warnings since we check above if the user cares
        vec = vec.copy()
        #if not np.any(norms >0.):
        #norms  = sum(norms)
        #print 'norms = ',type(norms),norms
        vec[:,0] /= norms
        vec[:,1] /= norms
        # np.seterr(**oldSettings)

    else:
        raise ValueError("I don't know how to normalize a vector array with > 2 dimensions")
    
    if return_mag:
        return vec, norms
    else:
        return vec

def triangle_area(node1,node2,node3):
    """
    area of a 2D triangular cell
    which is assumed to be ordered counter clockwise.
    
         1              2
          o------------o
           \         .
            \       .
             \    .
              \ .
               o
               3
    Note: Area vector is computed as the cross product of edge vectors [32] and [31].
    """
    q1 = node1.vector
    q2 = node2.vector
    q3 = node3.vector
    x1,x2,x3 = q1[0],q2[0],q3[0]
    y1,y2,y3 = q1[1],q2[1],q3[1]
    # area = -0.5*( (x1-x3)*(y2-y3)-(y1-y3)*(x2-x3) )      #<- cross product
    return  0.5*( x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2) ) #re-arranged

def triangle_area_from_raw_data(x1,x2,x3, y1,y2,y3):
    """
    area of a 2D triangular cell
    which is assumed to be ordered counter clockwise.
    
         1              2
          o------------o
           \         .
            \       .
             \    .
              \ .
               o
               3
    Note: Area vector is computed as the cross product of edge vectors [32] and [31].
    """
    # area = -0.5*( (x1-x3)*(y2-y3)-(y1-y3)*(x2-x3) )      #<- cross product
    return  0.5*( x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2) ) #re-arranged

'''
3D vector utilities
'''


## A wrapper for the purposes of this class, to avoid interacting with numpy
def Vector3D(x,y,z):
    return np.array([float(x),float(y),float(z)])

def printVec3(v):
    return "({:.5f}, {:.5f}, {:.5f})".format(v[0], v[1], v[2])



# Normalizes a numpy vector
# This methods modifies its argument in place, but also returns a reference to that
# array for chaining.
# Works on both single vectors and nx3 arrays of vectors (perfomed in-place).
# If zeroError=False, then this function while silently return a same-sized 0
# for low-norm vectors. If zeroError=True it will throw an exception
def normalize(vec, zeroError=False):

    # Used for testing zeroError
    eps = 0.00000000001

    # Use separate tests for 1D vs 2D arrays (TODO is there a nicer way to do this?)
    if(len(vec.shape) == 1):

        norm = np.linalg.norm(vec)
        if(norm < 0.0000001):
            if(zeroError):
                raise ArithmeticError("Cannot normalize function with norm near 0")
            else:
                vec[0] = 0
                vec[1] = 0
                vec[2] = 0
                return vec
        vec[0] /= norm
        vec[1] /= norm
        vec[2] /= norm
        return vec

    elif(len(vec.shape) == 2):

        # Compute norms for each vector
        norms = np.sqrt( vec[:,0]**2 + vec[:,1]**2 + vec[:,2]**2 )

        # Check for norm zero, if checking is enabled
        if(zeroError and np.any(norms < 0.00000000001)):
            raise ArithmeticError("Cannot normalize function with norm near 0")

        # Normalize in place
        # oldSettings = np.seterr(invalid='ignore')    # Silence warnings since we check above if the user cares
        vec[:,0] /= norms
        vec[:,1] /= norms
        vec[:,2] /= norms
        # np.seterr(**oldSettings)

    else:
        raise ValueError("I don't know how to normalize a vector array with > 2 dimensions")

    return vec


# Normalizes a numpy vector.
# This method returns a new (normalized) vector
# Works on both single vectors and nx3 arrays of vectors (perfomed in-place).
# If zeroError=False, then this function while silently return a same-sized 0
# for low-norm vectors. If zeroError=True it will throw an exception
def normalized(vec, zeroError=False):

    # Used for testing zeroError
    eps = 0.00000000001

    # Use separate tests for 1D vs 2D arrays (TODO is there a nicer way to do this?)
    if(len(vec.shape) == 1):

        norm = np.linalg.norm(vec)
        if(norm < 0.0000001):
            if(zeroError):
                raise ArithmeticError("Cannot normalize function with norm near 0")
            else:
                return np.zeros_like(vec)
        return vec / norm

    elif(len(vec.shape) == 2):

        # Compute norms for each vector
        norms = np.sqrt( vec[:,0]**2 + vec[:,1]**2 + vec[:,2]**2 )

        # Check for norm zero, if checking is enabled
        if(zeroError and np.any(norms < 0.00000000001)):
            raise ArithmeticError("Cannot normalize function with norm near 0")
        else:
            norms += 1.e-8
            #            if np.any(norms[:,0] < 0.00000000001):
            #                norms[:,0] += 1.e-8
            #            elif np.any(norms[:,1] < 0.00000000001):
            #                norms[:,1] += 1.e-8
            #            elif np.any(norms[:,2] < 0.00000000001):
            #                norms[:,2] += 1.e-8
        # Normalize in place
        # oldSettings = np.seterr(invalid='ignore')    # Silence warnings since we check above if the user cares
        vec = vec.copy()
        #if not np.any(norms >0.):
        #norms  = sum(norms)
        #print 'norms = ',type(norms),norms
        vec[:,0] /= norms
        vec[:,1] /= norms
        vec[:,2] /= norms
        # np.seterr(**oldSettings)

    else:
        raise ValueError("I don't know how to normalize a vector array with > 2 dimensions")

    return vec

def u2w(u, gamma = 1.4):
        '''
        Compute primitive variables from conservative variables.

        Parameters
        ----------
        u : conservative variables (rho, rho*u, rho*v, rho*E)

        Returns
        -------
        w : primitive variables (rho,     u,     v,     p)

        '''
        nq=4
        w = np.zeros((nq),float)
        
        ir = 0#self.ir
        iu = 1#self.iu
        iv = 2#self.iv
        ip = 3#self.ip
        
        
        #if u[0] == 0.0: 
        #    u[0] = 1.0e-15#1.e15
        #    #print('setting u density to 1e-15 to fix devide by zero in u2w')
        
        w[ir] = u[0]
        w[iu] = u[1]/u[0]
        w[iv] = u[2]/u[0]
        w[ip] = (gamma-1.0)*( u[3] - \
                              0.5*w[0]*(w[1]*w[1] + w[2]*w[2]) )
        return w
        

#******************************************************************************
# Printing Utilities
#
def default_input( message, defaultVal ):
    """http://stackoverflow.com/
    questions/5403138/how-to-set-a-default-string-for-raw-input
    """
    if defaultVal:
        return raw_input( "%s [%s]:" % (message,defaultVal) ) or defaultVal
    else:
        return raw_input( "%s " % (message) )