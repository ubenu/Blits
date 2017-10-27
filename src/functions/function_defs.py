'''
Created on 24 May 2017

@author: SchilsM
'''
import numpy as np

'''
Template for fn_x(x, p)
@x: a numpy array of shape (n_indi, n_points), 
    where n_indi is the number of independents, and
    n_points is the the number of points in each of the independents.
    Thus, x[0] is the first (and potentially only) independent.
    There should be at least one row for the independent, but currently
    this is not checked - potential crash point
@p: a numpy array of shape (n_par, ) containing the current
    variable parameter values
@return: the evaluated function values, a numpy array of shape (npoints, )

NOTE: the fn_x functions are used directly by the curve fitting routine
    and must be checked for integrity beforehand. 

Template for p0_fn_x(data, n_parameters)
@data: a numpy array of shape (n_indi+1, n_points), 
    where n_indi is the number of independents, and
    n_points is the the number of points in the observations.
    Thus, data[:-1] are the independents, and data[-1] is the dependent. 
@n_parameters: the number of parameter values for this function
@return p0: a numpy array of shape (n_par, ) containing an initial estimate
    for the variable parameter values
'''

def n_x(data):
    return data.shape[0] - 1

def data_valid(data, n_idependents):
    if data.shape[0] > 1:
        x = data[:-1]
        y = data[-1]
        return x.shape[1] == y.shape[0] and n_x(data) >= n_idependents
    return False
    
def fn_average(x, p):
    a, = p
    x0 = np.ones_like(x[0])
    return x0 * a

def p0_fn_average(data, n_parameters):
    n_independents = 1
    if data_valid(data, n_independents):
        p0 = np.ones((n_parameters,), dtype=float)
        return p0
    return None

def fn_straight_line(x, p):
    a, b = p
    x0 = x[0]
    return a + b * x0

def p0_fn_straight_line(data, n_parameters):
    n_independents = 1
    if data_valid(data, n_independents):
        p0 = np.ones((n_parameters,), dtype=float)
        return p0
    return None

def fn_1exp(x, p):
    a0, a1, k1 = p
    t = x[0]
    return a0 + a1*np.exp(-t*k1)

def p0_fn_1exp(data, n_parameters):
    n_independents = 1
    if data_valid(data, n_independents):
        x = data[:-1]
        y = data[-1]
        a0 = y[-1]
        a1 = y[0] - a0
        k1 = 1.0/(0.1 * x[0][-1] + np.finfo(float).tiny) # avoid division by zero
        p0 = np.array([a0, a1, k1])
        if p0.shape[0] == n_parameters:
            return p0
        return None
    return None    

def fn_1exp_strline(x, p):
    a0, a1, k1, b = p
    t = x[0]
    return a0 + a1*np.exp(-t*k1) + b * t

def p0_fn_1exp_strline(data, n_parameters):
    n_independents = 1
    if data_valid(data, n_independents):
        x = data[:-1]
        y = data[-1]
        a0 = y[-1]
        a1 = y[0] - a0
        k1 = 1.0/(0.1 * x[0][-1] + np.finfo(float).tiny) # avoid division by zero
        b = 0.0
        p0 = np.array([a0, a1, k1, b])
        if p0.shape[0] == n_parameters:
            return p0
        return None
    return None    
    
def fn_2exp(x, p):
    a0, a1, k1, a2, k2 = p
    t = x[0]
    return a0 + a1*np.exp(-t*k1) + a2*np.exp(-t*k2)

def p0_fn_2exp(data, n_parameters):
    n_independents = 1
    if data_valid(data, n_independents):
        x = data[:-1]
        y = data[-1]
        a0 = y[-1]
        a1 = y[0] - a0
        a2 = a1/2.0
        k1 = 1.0/(0.1 * np.abs(x[0][0] - x[0][-1]) + np.finfo(float).tiny) # avoid division by zero
        k2 = 0.1 * k1
        p0 = np.array([a0, a1, k1, a2, k2])
        if p0.shape[0] == n_parameters:
            return p0
        return None
    return None    
    
def fn_2exp_strline(x, p):
    a0, a1, k1, a2, k2, b = p
    t = x[0]
    return a0 + a1*np.exp(-t*k1) + a2*np.exp(-t*k2) + b * t

def p0_fn_2exp_strline(data, n_parameters):
    n_independents = 1
    if data_valid(data, n_independents):
        x = data[:-1]
        y = data[-1]
        a0 = y[-1]
        a1 = y[0] - a0
        a2 = a1/2.0
        k1 = 1.0/(0.1 * x[0][-1] + np.finfo(float).tiny) # avoid division by zero
        k2 = 0.1 * k1
        b = 0.0
        p0 = np.array([a0, a1, k1, a2, k2, b])
        if p0.shape[0] == n_parameters:
            return p0
        return None
    return None    
    
def fn_3exp(x, p):
    a0, a1, k1, a2, k2, a3, k3 = p
    t = x[0]
    return a0 + a1*np.exp(-t*k1) + a2*np.exp(-t*k2) + a3*np.exp(-t*k3)

def p0_fn_3exp(data, n_parameters):
    n_independents = 1
    if data_valid(data, n_independents):
        x = data[:-1]
        y = data[-1]
        a0 = y[-1]
        a1 = y[0] - a0
        a2 = a1
        a3 = a1
        k1 = 1.0/(0.1 * x[0][-1] + np.finfo(float).tiny) # avoid division by zero
        k2 = 0.1 * k1
        k3 = 0.01 * k1
        p0 = np.array([a0, a1, k1, a2, k2, a3, k3])
        if p0.shape[0] == n_parameters:
            return p0
        return None
    return None 

def fn_mich_ment(x, p):
    km, vmax = p
    s = x[0]
    return vmax * s / (km + s)

def p0_fn_mich_ment(data, n_parameters):
    n_independents = 1
    if data_valid(data, n_independents):
        x = data[:-1]
        y = data[-1]
        km = x.max()/2.0
        vmax = y.max()
        p0 = np.array([km, vmax])
        if p0.shape[0] == n_parameters:
            return p0
        return None
    return None  


def fn_comp_inhibition(x, p):
    km, ki, vmax = p
    s = x[0]
    i = x[1]
    return vmax * s / (km * (1.0 + i / ki) + s)
    
def p0_fn_comp_inhibition(data, n_parameters):
    n_independents = 2
    if data_valid(data, n_independents):
        x = data[:-1]
        y = data[-1]
        km = x.max()/2.0
        ki = km
        vmax = y.max()
        p0 = np.array([km, ki, vmax])
        if p0.shape[0] == n_parameters:
            return p0
        return None
    return None  
  
    
def fn_hill(x, p):
    ymax, xhalf, h = p
    x0 = x[0]
    return ymax / (np.power(xhalf/x0, h) + 1.0)

def p0_fn_hill(data, n_parameters):
    n_independents = 1
    if data_valid(data, n_independents):
        p0 = np.ones((n_parameters,), dtype=float)
        return p0
    return None
