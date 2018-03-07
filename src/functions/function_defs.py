'''
Created on 24 May 2017

@author: SchilsM
'''
import numpy as np
from scipy.optimize import brentq

'''
Template for fn_x(x, params)
@x: a numpy array of shape (n_indi, n_points), 
    where n_indi is the number of independents, and
    n_points is the the number of points in each of the independents.
    Thus, x[0] is the first (and potentially only) independent.
    There should be at least one row for the independent, but currently
    this is not checked - potential crash point
@params: a numpy array of shape (n_par, ) containing the current
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

def n_independents(data):
    return data.shape[0] - 1

def data_valid(data, n_idependents):
    if data.shape[0] > 1:
        x = data[:-1]
        y = data[-1]
        return x.shape[1] == y.shape[0] and n_independents(data) >= n_idependents
    return False
    
def fn_average(x, params):
    a, = params
    x0 = np.ones_like(x[0])
    return x0 * a

def p0_fn_average(data, n_parameters):
    n_independents = 1
    if data_valid(data, n_independents):
        x = data[:-1]
        y = data[-1]
        m = (y.min() + y.max()) / 2.0
        p0 = np.array([m, ], dtype=float)
        if p0.shape[0] == n_parameters:
            return p0
        return None
    return None

def fn_straight_line(x, params):
    a, b = params
    x0 = x[0]
    return a + b * x0

def p0_fn_straight_line(data, n_parameters):
    n_independents = 1
    if data_valid(data, n_independents):
        x = data[:-1][0]
        y = data[-1]
        b = (y[0] - y[-1]) / (x[0] - x[-1])
        a = y[0] - b * x[0]
        p0 = np.array([a, b], dtype=float)
        if p0.shape[0] == n_parameters:
            return p0
        return None
    return None

def fn_1exp(x, params):
    a0, a1, k1 = params
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

def fn_1exp_strline(x, params):
    a0, a1, k1, b = params
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
    
def fn_2exp(x, params):
    a0, a1, k1, a2, k2 = params
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
    
def fn_2exp_strline(x, params):
    a0, a1, k1, a2, k2, b = params
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
    
def fn_3exp(x, params):
    a0, a1, k1, a2, k2, a3, k3 = params
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

def fn_mich_ment(x, params):
    km, vmax = params
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


def fn_comp_inhibition(x, params):
    km, ki, vmax = params
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
  
def fn_uncomp_inhibition(x, params):
    km, ki, vmax = params
    s = x[0]
    i = x[1]
    return vmax * s / (km + s * (1.0 + i / ki))
    
def p0_fn_uncomp_inhibition(data, n_parameters):
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
  
def fn_noncomp_inhibition(x, params):
    km, ki, vmax = params
    s = x[0]
    i = x[1]
    return vmax * s / ((km + s) * (1.0 + i / ki))
    
def p0_fn_noncomp_inhibition(data, n_parameters):
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
  
def fn_mixed_inhibition(x, params):
    km, ki, kis, vmax = params
    s = x[0]
    i = x[1]
    return vmax * s / (km * (1.0 + i / ki) + s * (1.0 + i / kis))
    
def p0_fn_mixed_inhibition(data, n_parameters):
    n_independents = 2
    if data_valid(data, n_independents):
        x = data[:-1]
        y = data[-1]
        km = x.max()/2.0
        ki = km
        kis = km
        vmax = y.max()
        p0 = np.array([km, ki, kis, vmax])
        if p0.shape[0] == n_parameters:
            return p0
        return None
    return None  
  
    
def fn_hill(x, params):
    ymax, xhalf, h = params
    x0 = x[0]
    return ymax / (np.power(xhalf/x0, h) + 1.0)

def p0_fn_hill(data, n_parameters):
    n_independents = 1
    if data_valid(data, n_independents):
        p0 = np.ones((n_parameters,), dtype=float)
        return p0
    return None

def fn_comp_binding(x, params):
    KdL, KdN, eP, eL, eN, ePL, ePN = params
    p0 = x[0]
    l0 = x[1]
    n0 = x[2]
    p = np.empty_like(p0)
    fn = root_fn_comp_bind
    i = 0
    for p0i, l0i, n0i in zip(p0, l0, n0):
        p[i] = brentq(fn, 0, p0i, (KdL, KdN, p0i, l0i, n0i)) 
        i += 1        
    pl = l0*p/(KdL+p)
    pn = n0*p/(KdN+p)
    l = l0 - pl
    n = n0 - pn
    return eL*l + eN*n + eP*p + ePL*pl + ePN*pn

def root_fn_comp_bind(p, Kdl, Kdn, p0, l0, n0):
    result = p*p*p + (Kdl+Kdn+l0+n0-p0)*p*p + (Kdl*Kdn+l0*Kdn+n0*Kdl-(Kdl+Kdn)*p0)*p - Kdl*Kdn*p0
    return result #p * ( 1 + l0/(Kdl+p) + n0/(Kdn+p) ) - p0

def p0_fn_comp_binding(data, n_parameters):
    n_independents = 3
    if data_valid(data, n_independents):
        p0 = np.ones((n_parameters,), dtype=float)
        return p0
    return None

def fn_chem_unfold(x, params):
    """
Let DeltaG0NU = A(1): Let mNU = A(2): Let IntN = A(3): Let SlopeN = A(4)
Let IntU = A(5): Let SlopeU = A(6): Let TempC = A(7)

Let RT = 0.0019872 * (TempC + 273)
Let DeltaGNU = DeltaG0NU + mNU * Denat
Let KNU = Exp(-DeltaGNU / RT)
Let FractionN = 1 / (1 + KNU)
Let FractionU = 1 - FractionN
   
Let CalculatedY = (IntN + SlopeN * Denat) * FractionN + (IntU + SlopeU * Denat) * FractionU
 
    """
    dG0NU, mNU, intN, slopeN, intU, slopeU, tempC = params
    denat = x[0]
    rt = 0.0019872 * (tempC + 273.)
    dGNU = dG0NU + mNU * denat
    kNU = np.exp(-dGNU / rt)
    frN = 1. / (1. + kNU)
    frU = 1. - frN
    return (intN + slopeN * denat) * frN + (intU + slopeU * denat) * frU
    
def p0_fn_chem_unfold(data, n_parameters):
    n_independents = 1
    if data_valid(data, n_independents):
        y = data[-1]
        dG0NU = 1.0
        mNU = -1. 
        intN = y[0] 
        slopeN = 0.0 
        intU = y[-1] 
        slopeU = 0.0 
        tempC = 25.0
        p0 = np.array([dG0NU, mNU, intN, slopeN, intU, slopeU, tempC])
        return p0
    return None

def fn_therm_unfold(x, params):
    """
Let DeltaHmNU = A(1): Let TmNU = A(2): Let IntN = A(3):
Let SlopeN = A(4): Let IntU = A(5): Let SlopeU = A(6): Let DeltaCpNU = A(7)
 
Let RT = 0.0019872 * (TempC + 273)
Let DeltaGNU = DeltaHmNU * (1 - (273 + TempC) / (TmNU + 273)) + DeltaCpNU * (TempC - TmNU - (TempC + 273) * Log((TempC + 273) / (TmNU + 273)))
Let KNU = Exp(-DeltaGNU / RT)
Let FractionN = 1 / (1 + KNU)
FractionU = 1 - FractionN
   
Let CalculatedY = (IntN + SlopeN * TempC + extra * TempC * TempC) * FractionN + (IntU + SlopeU _
* TempC) * FractionU
    """
    dHmNU, tmNU, intN, slopeN, intU, slopeU, dCpNU = params
    tempC = x[0]
    tempK = tempC + 273.
    tmKNU = tmNU + 273.
    rt = 0.0019872 * tempK
    dGNU = dHmNU * (1. - tempK/tmKNU) + dCpNU * (tempK - tmKNU - tempK * np.log(tempK / tmKNU))
    kNU = np.exp(-dGNU / rt)
    frN = 1. / (1. + kNU)
    frU = 1. - frN
    return (intN + slopeN * tempC) * frN + (intU + slopeU * tempC) * frU

def p0_fn_therm_unfold(data, n_parameters):
    n_independents = 1
    if data_valid(data, n_independents):
        x = data[:-1]
        y = data[-1]
        dHmNU = 1.0
        tmNU = (x[0][-1] + x[0][0]) / 2.0 
        intN = y[0] 
        slopeN = 0.0 
        intU = y[-1] 
        slopeU = 0.0 
        dCpNU = 0.0
        p0 = np.array([dHmNU, tmNU, intN, slopeN, intU, slopeU, dCpNU])
        return p0
    return None
    
    
