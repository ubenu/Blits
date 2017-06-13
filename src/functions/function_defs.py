'''
Created on 24 May 2017

@author: SchilsM
'''
import numpy as np

def fn_average(x, p):
    a, = p
    return a

def p0_fn_average(x, y, p):
    return np.ones_like(p)

def fn_straight_line(x, p):
    a, b = p
    return a + b*x

def p0_fn_straight_line(x, y, p):
    a, b = p
    return np.ones_like(p)

def fn_1exp(x, p):
    a0, a1, k1 = p
    t = x
    return a0 + a1*np.exp(-t*k1)

def p0_fn_1exp(x, y, p):
    a0, a1, k1 = p
    t = x
    a0 = y[-1]
    a1 = y[0] - a0
    return 
    
def fn_2exp(x, p):
    a0, a1, k1, a2, k2 = p
    t = x
    return a0 + a1*np.exp(-t*k1) + a2*np.exp(-t*k2)
    
def fn_3exp(x, p):
    a0, a1, k1, a2, k2, a3, k3 = p
    t = x
    return a0 + a1*np.exp(-t*k1) + a2*np.exp(-t*k2) + a3*np.exp(-t*k3)

def fn_mich_ment(x, p):
    km, vmax = p
    s = x
    return vmax * s / (km + s)

def fn_comp_inhibition(x, p):
    km, ki, vmax = p
    s = x[0]
    i = x[1]
    return vmax * s / (km * (1.0 + i / ki) + s)
    
def fn_hill(x, p):
    ymax, xhalf, h = p
    return ymax / (np.power(xhalf/x, h) + 1.0)