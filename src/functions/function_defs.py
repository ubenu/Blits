'''
Created on 24 May 2017

@author: SchilsM
'''
import numpy as np

def fn_average(x, p):
    a, = p
    x0 = np.ones_like(x[0])
    return x0 * a

def p0_fn_average(x, y):
    p0 = np.array([y.mean()])
    return p0

def fn_straight_line(x, p):
    a, b = p
    x0 = x[0]
    return a + b * x0

def p0_fn_straight_line(x, y):
    p0 = np.ones((2,), dtype=float)
    return p0

def fn_1exp(x, p):
    a0, a1, k1 = p
    t = x[0]
    return a0 + a1*np.exp(-t*k1)

def p0_fn_1exp(x, y):
    a0 = y.iloc[-1]
    a1 = y.iloc[0] - a0
    k1 = 1.0/(0.1 * x.iloc[-1])
    p0 = np.array([a0, a1, k1])
    return p0

def fn_1exp_strline(x, p):
    a0, a1, k1, b = p
    t = x[0]
    return a0 + a1*np.exp(-t*k1) + b * t

def p0_fn_1exp_strline(x, y):
    a0 = y.iloc[-1]
    a1 = y.iloc[0] - a0
    k1 = 1.0/(0.1 * x.iloc[-1])
    b = 0.0
    p0 = np.array([a0, a1, k1, b])
    return p0
    
    
def fn_2exp(x, p):
    a0, a1, k1, a2, k2 = p
    t = x[0]
    return a0 + a1*np.exp(-t*k1) + a2*np.exp(-t*k2)

def p0_fn_2exp(x, y):
    a0 = y.iloc[-1]
    a1 = y.iloc[0] - a0
    k1 = 1.0/(0.1 * x.iloc[-1])
    a2 = a1/2.0
    k2 = 1.0/(1.0 * x.iloc[-1])
    p0 = np.array([a0, a1, k1, a2, k2])
    return p0
    
def fn_2exp_strline(x, p):
    a0, a1, k1, a2, k2, b = p
    t = x[0]
    return a0 + a1*np.exp(-t*k1) + a2*np.exp(-t*k2) + b * t

def p0_fn_2exp_strline(x, y):
    a0 = y.iloc[-1]
    a1 = y.iloc[0] - a0
    k1 = 1.0/(0.1 * x.iloc[-1])
    a2 = a1/2.0
    k2 = 1.0/(1.0 * x.iloc[-1])
    b = 0.0
    p0 = np.array([a0, a1, k1, a2, k2, b])
    return p0
    
def fn_3exp(x, p):
    a0, a1, k1, a2, k2, a3, k3 = p
    t = x[0]
    return a0 + a1*np.exp(-t*k1) + a2*np.exp(-t*k2) + a3*np.exp(-t*k3)

def p0_fn_3exp(x, y):
    a0 = y.iloc[-1]
    a1 = (y.iloc[0] - a0)/3
    k1 = 1.0/(0.1 * x.iloc[-1])
    a2 = a1
    k2 = 1.0/(1.0 * x.iloc[-1])
    a3 = a1
    k3 = 1.0/(10.0 * x.iloc[-1])
    p0 = np.array([a0, a1, k1, a2, k2, a3, k3])
    return p0

def fn_mich_ment(x, p):
    km, vmax = p
    s = x[0]
    return vmax * s / (km + s)

def p0_fn_mich_ment(x, y):
    km = x.iloc[-1]/2.0
    vmax = y.max()
    p0 = np.array([km, vmax])
    return p0

def fn_comp_inhibition(x, p):
    km, ki, vmax = p
    s = x[0]
    i = x[1]
    return vmax * s / (km * (1.0 + i / ki) + s)
    
def p0_fn_comp_inhibition(x, y):
    p0 = np.ones((3,), dtype=float)
    return p0    
    
def fn_hill(x, p):
    ymax, xhalf, h = p
    x0 = x[0]
    return ymax / (np.power(xhalf/x0, h) + 1.0)

def p0_fn_hill(x, y):
    p0 = np.ones((3,), dtype=float)
    return p0
