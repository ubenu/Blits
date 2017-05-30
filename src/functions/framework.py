'''
Created on 24 May 2017

@author: SchilsM
'''

import numpy as np
from scipy.optimize import curve_fit
import functions.function_defs as fdefs
#from functions import function_defs
#from statsmodels.nonparametric.kernels import d_gaussian

def make_func(fn, x, params, const):
    """
    @params is the full input array for fn
    @const is a disctionary with the indices and values of the papameters
    that need to be kept constant
    """
    mask = np.ones((len(params),), dtype=bool)
    for index in const:
        params[index] = const[index]
        mask[index] = False
    def func(x, *v):
        params[mask] = v 
        return fn(x, params)
    return func

def make_func_global(fn, x, links):
    """
    @x is a 2 or higher dimensional array of shape (n_curves, n_x^+) 
    (^+ indicates, there must be one or more such dimension)
    @links is a 2D array of shape (n_curves, n_params_for_fn) that contains
    the index of its associated (unique) parameter in v 
    """
    p_shape = links.shape
    links = links.flatten()
    def func(x, *v):
        params = np.zeros(p_shape, dtype=float).flatten()
        for i in range(params.shape[0]):
            params[i] = v[links[i]]
        params = params.reshape(p_shape)
        y = np.zeros(x.shape, dtype=float)
        for i in range(x.shape[0]):
            y[i] = fn(x[i], params[i])
        y_out = y.flatten()
        return y_out
    return func      
    
def test_global():
    import matplotlib.pyplot as plt
    # Make the independent axes
    n_x0, n_x1 = 7, 4 
    x0 = np.ones((n_x1, n_x0))
    x0_template = np.array([1,2,4,6,10,15,20], dtype=float)
    x0 = (x0 * x0_template).flatten()
    x1 = np.ones((n_x0, n_x1))
    x1_template = np.array([0,5,10,20], dtype=float)
    x1 = (x1 * x1_template).transpose().flatten()
    x = np.vstack((x0, x1))
    x = x.reshape((x.shape[0], n_x1, n_x0))
    for x_i in x:
        print(x_i)
    km = 2.0
    ki = 5.0
    vm = 100.
     
    
    
    n_curves = 5
    n_points = 8
    x_dim = 2
    x0_template = np.array([1,2,4,6,10,15,20], dtype=float)
    x0 = np.ones((n_curves))
    x_start, x_end = 2.5, 50
    std = 0.02
    fn = fdefs.fn_mich_ment
    d_shape = (n_curves, x_dim, n_points)
    x = np.linspace(x_start, x_end, n_points)
    y = np.ones(d_shape)
    km = np.ones((n_curves)) * 2.5
    ki = np.array([1.5, 3.8, 5.7, 7.2, 9.1])
    vm = np.ones((n_curves)) * 30.0
    count = 0
    for curve in y:
        params = np.array([km[count], ki[count], vm[count]])
        y[count] = curve * fn(x, params)
        count += 1 
    noisy_y = y + np.random.normal(0.0, std * y.max(), y.shape)
    
#     # non-global fit (potentially with constants)
#     fit_y = np.zeros(d_shape)
#     count = 0
#     for curve in noisy_y:
#         p = np.ones((2,))
#         c = {}
#         m = [not (i in c) for i in range(p.shape[0])]
#         p_est = np.ones(p[m].shape[0], dtype=float)
#         curve_fit(make_func(function_defs.fn_mich_ment, x, p, c), x, curve, p0=p_est)
#         fit = fn(x, p)
#         fit_y[count] = fit
#         count += 1
#     
    
    # global fit
    gfit_x = (np.ones(d_shape) * x)
    n_params = 2 # we're using fn_mich_ment
    glinks = np.array([0, 1, 2, 1, 3, 1, 4, 1, 5, 1])
    glinks = glinks.reshape((n_curves, n_params))
    p_est = np.zeros(np.unique(glinks).shape)
    
    out = curve_fit(make_func_global(fn, x, glinks), gfit_x, noisy_y.flatten(), p0=p_est)
    print(out[0])
    p_shape = glinks.shape
    glinks = glinks.flatten()
    params = np.zeros(p_shape).flatten()
    fv = out[0]
    for i in range(params.shape[0]):
        params[i] = fv[glinks[i]]
    params = params.reshape(p_shape)
    print(params)
    count = 0
    gfit_y = np.zeros_like(gfit_x)
    for x_i in gfit_x:
        p_i = params[count]
        gfit_y[count] = fn(x_i, p_i)
        count += 1
        
    # Lineweaver-Burke
    plt.plot(gfit_x.transpose(), noisy_y.transpose(),'ro', gfit_x.transpose(), gfit_y.transpose(), 'k-')
    plt.show()
         
    
    
def test():   
    import matplotlib.pyplot as plt
    
    x = np.arange(0, 1, 0.001)
    f = fdefs.fn_3exp
    p = np.array([1.0, 0.5, 10.0, 0.25, 1.0, 0.0, 1.0])
    c = {5: 0.0, 6: 1.0}
    m = [not (i in c) for i in range(len(p))]
    p_est = np.ones(len(p[m]), dtype=float)
    data = f(x, p)
    ndata = data + np.random.normal(0.0, 0.02*data, len(data))
    
    curve_fit(make_func(f, x, p, c), x, ndata, p0=p_est)
    # as p is a reference, variable params gets changed by curve_fit; no need to 
    # replace anything in p
    
    print(p)
    plt.plot(x, ndata, 'ro', x, f(x, p), 'k-')
    plt.show()

if __name__ == '__main__':
    test_global()




    

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