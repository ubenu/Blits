'''
Created on 24 May 2017

@author: SchilsM
'''

import numpy as np
from scipy.optimize import curve_fit
import functions.function_defs as fdefs
#from functions import function_defs
#from statsmodels.nonparametric.kernels import d_gaussian

def make_func(fn, params, const={}):
    """
    @fn is a function with signature fn(x, p), where p is a 1D array of parameters
    @x is a (k, m) shaped array, where k is the number of independents and m is the number of points
    @params is the full input array for fn; same length as p (fn argument)
    @const is a dictionary with the indices (keys) and values (values) of the parameters
    that must be kept constant
    """
    mask = np.ones((len(params),), dtype=bool)
    for index in const:
        params[index] = const[index]
        mask[index] = False
    def func(x, *v):
        params[mask] = v 
        return fn(x, params)
    return func


def make_func_global(fn, x_splits, links, consts={}):
    """
    @fn is a function with signature fn(x, p), where p is a 1D array of parameters
    @x is a (k, m) shaped array, where k is the number of independents and m is the number of points
    @links is an array of shape (#curves, #params_for_fn) that contains
    the index of its associated (unique) parameter used to construct v
    @consts is a dictionary with the indices (keys) and values (values) of the parameters
    that must be kept constant
    """
    n_curves = links.shape[0]
    n_params = links.shape[1]
    links = links.flatten()
    def func(x, *v):
        split_x = np.split(x, x_splits, axis=1)
        if len(split_x) != n_curves:
            print("Error: incorrect x split in make_func_global")
        params = np.zeros_like(links, dtype=float)
        u = np.zeros_like(np.unique(links), dtype=float) 
        # u is the array that will contain the values of all unique params
        # v is the array of variable params
        mask = np.ones_like(u, dtype=bool)
        for i in consts:
            u[i] = consts[i]
            mask[i] = False
        u[mask] = v #np.array(v)[mask]
        for i in range(links.shape[0]):
            params[i] = u[links[i]] # fill in the parameter values
        params = params.reshape((n_curves, n_params))
        y_out = []
        for i in range(n_curves):
            y_out.extend(fn(split_x[i], params[i]))
        return y_out
    return func      
    
def test_global():
    import matplotlib.pyplot as plt
    # Create the independent (x)
    fn = fdefs.fn_comp_inhibition # has 3 params: km, ki, vm, in that order  
    km = 2.0
    ki = 5.0
    vm = 100.
    p = np.array([km, ki, vm])
    
    n_points, n_curves = 7, 4 
    x0 = np.ones((n_curves, n_points))
    x0_template = np.array([1,2,4,6,10,15,20], dtype=float)
    x0 = (x0 * x0_template).flatten()
    x1 = np.ones((n_points, n_curves))
    x1_template = np.array([0,5,10,20], dtype=float)
    x1 = (x1 * x1_template).transpose().flatten()
    x = np.vstack((x0, x1)).transpose()
    xs = np.split(x, np.arange(n_points, n_points*n_curves, n_points), axis=0)
    x = []
    for x_i in xs:
        x.append(x_i.transpose())
    
    # Create the dependent 
    std = 0.02
    y = []
    for x_i in x:
        y_i = fn(x_i, p) # compute the value
        y_i = y_i + np.random.normal(0.0, std * y_i.max(), y_i.shape) # add some noise
        y.append(y_i)
    ay = np.array(y)

    # Do a fit to the individual curves using a different model
    ffn = fdefs.fn_comp_inhibition #fdefs.fn_mich_ment
    npars = 3
    fit_y, fit_p = [], []
    fp_in = np.ones((npars,))
    for i in range(len(x)):
        fp0 = np.ones_like(fp_in)
        c={1:5.0}
        m = [not (i in c) for i in range(fp0.shape[0])]
        p_est = np.ones(fp0[m].shape, dtype=float)
        curve_fit(make_func(ffn, fp_in, const=c), x[i], y[i], p0=p_est)
        fit_y.append(ffn(x[i], fp_in))
        fit_p.append(fp_in.copy())
    print(np.array(fit_p))
    afit_y = np.array(fit_y)
        
    # Do a global fit
    l_shape = (n_curves, npars)
    links = np.zeros(l_shape, dtype=int)
    links[:,1] += 1
    links[:,2] += 2 # all 3 params are linked in all curves (here)
#    links = np.array([0,1,2,1,3,1,4,1]).reshape(l_shape)
    #links = np.arange(n_curves * fp0.shape[0]).reshape(l_shape) # all params independent
    splits = np.arange(n_points, n_curves * n_points, n_points)
    
    flat_x = x[0]
    for x_i in x[1:]:
        flat_x = np.hstack((flat_x, x_i))
    flat_y = y[0]
    for y_i in y[1:]:
        flat_y = np.concatenate((flat_y, y_i))
    ups = np.unique(links)
    c = {1:5.}
    m = [not (i in c) for i in ups]
    p0 = np.ones_like(ups, dtype=float)[m]
    out = curve_fit(make_func_global(ffn, splits, links, consts=c), flat_x, flat_y, p0=p0)
    p_fin = np.zeros_like(ups, dtype=float)
    p_fin[m] = out[0]
    for i in c:
        p_fin[i] = c[i]
    print(p_fin)
    p_shape = links.shape
    flinks = links.flatten()
    fparams = np.zeros_like(flinks, dtype=float)
    for i in range(fparams.shape[0]):
        fparams[i] = p_fin[flinks[i]]
    fparams = fparams.reshape(p_shape)
    gfit_y = []
    for i in range(len(x)):
        gfit_y.append(ffn(x[i], fparams[i]))
    agfit_y = np.array(gfit_y)

    plt.plot(x[0][0], ay.transpose(), 'bo', x[0][0], afit_y.transpose(), 'k-', x[0][0], agfit_y.transpose(), 'r-') 
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