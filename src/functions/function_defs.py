'''
Created on 24 May 2017

@author: SchilsM
'''
import numpy as np

def fn_straight_line(x, p):
    a, b = p
    return (a + b*x)

def fn_1exp(x, p):
    a0, a1, k1 = p
    return (a0 + a1*np.exp(-x*k1))
    
def fn_2exp(x, p):
    a0, a1, k1, a2, k2 = p
    return (a0 + a1*np.exp(-x*k1) + a2*np.exp(-x*k2))
    
def fn_3exp(x, p):
    a0, a1, k1, a2, k2, a3, k3 = p
    return (a0 + a1*np.exp(-x*k1) + a2*np.exp(-x*k2) + a3*np.exp(-x*k3))

def fn_mich_ment(x, p):
    km, vmax = p
    return vmax * x / (km + x)
    
def fn_hill(x, p):
    ymax, xhalf, h = p
    return ymax / (np.power(xhalf/x, h) + 1.0)