'''
Created on 24 May 2017

@author: SchilsM
'''

import numpy as np
import scipy.optimize
import functions.function_defs

def fn_1exp(self, x, *p):
    a0, a1, k1 = p
    return (a0 + a1*np.exp(-x*k1))


def make_func(fn, x, y, params):
    c, v = [], []
    for p in params:
        if params[p][0]:
            c.append((p, params[p][1]))
        else:
            v.append((p, params[p][1]))
    return c, v


ps = {'a0':(False, 10.), 'a1':(True, 1.0), 'k1':(False, 1.5)}
func = fn_1exp
print(make_func(func, 0, 0, ps))


    

# Parameters can be 1) fixed, 2) variable, shared, 3) variable, not shared
# Fixed parameters should be fed to the fitting function separately, 
# eg in a make_func function, such as this one:
#
# def make_calc_scatter(self, constants):
#     p = constants['power']
#     def calc_scatter(x, *variables):
#         a0, c = variables
#         return a0 + np.power(c / x, p)
#     return calc_scatter

# Example fn:
# def fn_3exp(self, x, *p):
#     a0, a1, k1, a2, k2, a3, k3 = p
#     return (a0 + a1*np.exp(-x*k1) + a2*np.exp(-x*k2) + a3*np.exp(-x*k3))
#
# *p needs to be decomposed into *c (list of constants) and *v (list of variables)

# Suppose p[0], p[2] and p[3] (defined via UI) are constant
# 
#
#

# def generic_func(func, x, *p):
#     return func(x, *p)
# 
# def err_func(func, x, y, *p):
#     y_calc = generic_func(func, x, *p)
#     return(y_calc - y)



# def get_scatter_params(self, x_lo, x_hi, constants, params0):
#     d = copy.deepcopy(self.working_data[['X','Y1']])
#     dl = d[d['X'] >= x_lo]
#     d = dl[dl['X'] <= x_hi]
#     try:
#         params, covar = curve_fit(self.make_calc_scatter(constants), 
#                                   d['X'], d['Y1'], p0=params0)
#     except Exception as e:
#         print(e)
#         
#     return params
# 



"""
Jonathan J. Helmus jjhelmus@gmail.... 
Wed Apr 3 14:36:17 CDT 2013
Previous message: [SciPy-User] Nonlinear fit to multiple data sets with a shared parameter, and three variable parameters.
Next message: [SciPy-User] Nonlinear fit to multiple data sets with a shared parameter, and three variable parameters.
Messages sorted by: [ date ] [ thread ] [ subject ] [ author ]
Troels,

    Glad to see another NMR jockey using Python.  
    I put together a quick and dirty script showing how to do a global fit using Scipy's leastsq function.  
    Here I am fitting two decaying exponentials, first independently, 
    and then using a global fit where we require that the both trajectories have the same decay rate.  
    You'll need to abstract this to n-trajectories, but the idea is the same.  
    If you need to add simple box limit you can use leastsqbound (https://github.com/jjhelmus/leastsqbound-scipy) 
    for Scipy like syntax or Matt's lmfit for more advanced contains and parameter controls.  
    Also you might be interested in nmrglue (nmrglue.com) for working with NMR spectral data.

Cheers,

    - Jonathan Helmus




def sim(x, p):
    a, b, c  = p
    return np.exp(-b * x) + c

def err(p, x, y):
    return sim(x, p) - y


# set up the data
data_x = np.linspace(0, 40, 50)
p1 = [2.5, 1.3, 0.5]       # parameters for the first trajectory
p2 = [4.2, 1.3, 0.2]       # parameters for the second trajectory, same b
data_y1 = sim(data_x, p1)
data_y2 = sim(data_x, p2)
ndata_y1 = data_y1 + np.random.normal(size=len(data_y1), scale=0.01)
ndata_y2 = data_y2 + np.random.normal(size=len(data_y2), scale=0.01)

# independent fitting of the two trajectories
print ("Independent fitting")
p_best, ier = scipy.optimize.leastsq(err, p1, args=(data_x, ndata_y1))
print ("Best fit parameter for first trajectory: " + str(p_best))

p_best, ier = scipy.optimize.leastsq(err, p2, args=(data_x, ndata_y2))
print ("Best fit parameter for second trajectory: " + str(p_best))

# global fit

# new err functions which takes a global fit
def err_global(p, x, y1, y2):
    # p is now a_1, b, c_1, a_2, c_2, with b shared between the two
    p1 = p[0], p[1], p[2]
    p2 = p[3], p[1], p[4]
    
    err1 = err(p1, x, y1)
    err2 = err(p2, x, y2)
    return np.concatenate((err1, err2))

p_global = [2.5, 1.3, 0.5, 4.2, 0.2]    # a_1, b, c_1, a_2, c_2
p_best, ier = scipy.optimize.leastsq(err_global, p_global, 
                                    args=(data_x, ndata_y1, ndata_y2))

p_best_1 = p_best[0], p_best[1], p_best[2]
p_best_2 = p_best[3], p_best[1], p_best[4]
print ("Global fit results")
print ("Best fit parameters for first trajectory: " + str(p_best_1))
print ("Best fit parameters for second trajectory: " + str(p_best_2))
"""