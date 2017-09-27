'''
Created on 24 May 2017

@author: SchilsM
'''

import sys
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import distributions # t
import functions.function_defs as fdefs
#from astropy.modeling.tests.test_projections import pars
#from functions import function_defs
#from statsmodels.nonparametric.kernels import d_gaussian

class ModellingFunction():
    
    def __init__(self, name, fn_ref, p0_fn_ref, param_names, fn_str):
        self.id = id
        self.name = name
        self.fn_ref = fn_ref
        self.p0_fn_ref = p0_fn_ref
        self.param_names = param_names
        self.fn_str = fn_str
    
    def __str__(self):
        return self.fn_str
    

class FunctionsFramework():     
                  
    def __init__(self):
        pass

    def display_curve(self, fn, x, params):
        """
        @fn is a function with signature fn(x, p), where p is a 1D array of parameters
        @x is a (k, m) shaped array, where k is the number of independents and m is the number of points
        @paramsis the full input array for fn; same length as p (fn argument)
        """
        return fn(x, params)
        
    def confidence_intervals(self, n, params, covar, conf_level):
        """
        @n is the number of data points used for the estimation of params and covar
        @params is a 1D numpy array of best-fit parameter values
        @covar is the best fit covariance matrix
        @conf_level is the required confidence level, eg 0.95 for 95% confidence intervals
        Returns a 1D numpy array of the size of params with the confidence intervals
        on params (report (eg) p +/- d or p (d/p*100 %), where p is a parameter value
        and d is its associated relative confidence interval.
        """
        dof = max(0, n - params.shape[0]) # degrees of freedom
        tval = distributions.t.ppf(conf_level / 2., dof) # student-t value for the dof and confidence level
        sigma = np.power(np.diag(covar), 0.5) # standard error
        return sigma * tval
            
#     def make_func(self, fn, params, const={}):
#         """
#         @fn is a function with signature fn(x, p), where p is a tuple of parameter values
#         @x is a (k, m) shaped array, where k is the number of independents and m is the number of points
#         @params is the full input array for fn; same length as p (fn argument)
#         @const is a dictionary with the indices (keys) and values (values) of the parameters
#         that must be kept constant
#         """
#         filter_in = np.ones((len(params),), dtype=bool)
#         for index in const:
#             params[index] = const[index]
#             filter_in[index] = False
#         def func(x, *v):
#             params[filter_in] = v 
#             return fn(x, params)
#         return func#     

    def make_func(self, fn, params, variables):
        """
        @fn is a function with signature fn(x, *p), where *p is a tuple of parameter values
        @x is a (k, m) shaped array, where k is the number of independents and m is the number of points
        @params is the full input array for fn; same length as p (fn argument)
        @variables is an (n_params)-shaped array of Boolean values (parallel to @params):
        if True, parameter values is variable, if False, parameter value is to be kept constant.
        """
        def func(x, *v):
            params[variables] = v
            return fn(x, params)
        return func
    
    def make_func_global(self, fn, x_splits, param_vals, variables, groups):
        """
        @fn is a function with signature fn(x, p), where @p is 
        a tuple of parameter values and @x is a (k, m) shaped array, 
        with k the number of independents and m the total number of 
        data points (all curves concatenated along the array's 2nd axis)
        @x_splits is  is a sorted 1-D array of integers, whose entries 
        indicate where the array's second axis must be split to generate
        the individual curves.
        @param_vals is an (n_curves, n_params)-shaped array with values for
        each parameter in each curve.
        @variables is an (n_curves, n_params)-shaped array of Boolean values 
        (with rows and columns parallel to @curve_names and @param_names, respectively):
        if True, parameter values is variable, if False, parameter value is constant.
        @groups is an array of shape (n_curves, n_params) of integers, 
        containing the indices of the actual parameters to be fitted, 
        where n_curves is the number of curves and n_params the number of
        parameters taken by fn (ie len(p)).
        Example for 4 curves and 3 parameters:
              p0    p1    p2
        c0    0     2     3
        c1    0     2     4
        c2    1     2     5
        c3    1     2     6
        means that parameter p0 is assumed to have the same value in 
        curves c0 and c1, and in curves c2 and c3 (a different value), 
        and that the value for p1 is the same in all curves, whereas
        the value of p2 is different for all curves. In this example, 
        the total number of parameters to be fitted is 7.
        
        @returns [0] a function that can be used as input to curve_fit
        @returns [1] an initial estimate for the variable parameters
        """
        pshape = param_vals.shape
        ugroups, indices, inverse_indices = np.unique(groups.flatten(), 
                                                      return_index=True, 
                                                      return_inverse=True)
        # ugroups are the IDs of the unique groups in a flattened array
        # indices are the indices of the first occurrence of a unique group 
        #    in the flattened array or in a parallel one
        # reverse_indices indicates where to find a particular ID in ugroups 
        #    to reconstruct the original flat array (or a parallel one)
        uparams = param_vals.flatten()[indices]
        uv_filter = variables.flatten()[indices]
        def func(x, *v):
            split_x = np.split(x, x_splits, axis=1)
            uparams[uv_filter] = v
            params = uparams[inverse_indices].reshape(pshape)          
            y_out = []
            for x, p in zip(split_x, params):
                i_out = fn(x, p)
                y_out.extend(i_out)
            return y_out      
        return func, uparams[uv_filter] 
    
    def perform_fit(self, data, func, param_values, keep_constant, groups=None):
        if groups is None:
            return self.perform_standard_curve_fit(data, func, param_values, keep_constant)
        else:
            return self.perform_global_curve_fit(data, func, param_values, keep_constant, groups)
    
    def perform_standard_curve_fit(self, data, func, param_values, keep_constant):
        process_log = ""
        variables = np.logical_not(keep_constant)

        if np.any(variables): # There must be something to fit
            # Get the correct function for global fitting and a first estimate for the variable params
            #gfunc, p_est = self.make_func_global(func, x_splits, param_values, variables, groups)
            pass

        # Now fit all curves independently (as in Blivion)
        i = 0
        for curve in data:
            x, y = curve[:-1], curve[-1]
            p_est = param_values[i]
            var = variables[i]
            fnc = func  # creating fnc is probably unnecessary (func can be set to the func argument), but just in case
            if not np.all(var):                
                fnc = self.make_func(func, p_est, var)
                
            i += 1
            
            ftol, xtol = 1.0e-9, 1.0e-9
            pars = None
            while pars is None and ftol < 1.0:
                ftol *= 10.
                xtol *= 10.
                try:
                    out = curve_fit(fnc, x, y, p0=p_est, ftol=ftol, xtol=xtol, maxfev=250, full_output=1) 
                    pars = out[0]
                    #covar = out[1]
                    nfev = out[2]['nfev']    
                    log_entry = "\nTrace: " + '{:d}'.format(i) + "\tNumber of evaluations: " + '{:d}'.format(nfev) + "\tTolerance: " + '{:.1e}'.format(ftol)
                    process_log += log_entry
                    print(log_entry)
                    print(pars)
                except ValueError as e:
                    log_entry = "\nValue Error (ass):" + str(e)
                    process_log += log_entry
                    print(log_entry)
                except RuntimeError as e:
                    log_entry = "\nRuntime Error (ass):" + str(e)
                    process_log += log_entry
                    print(log_entry)
                except:
                    log_entry = "\nOther error (ass)"
                    process_log += log_entry
                    print(log_entry) 
            print(pars) 
            return pars
                      
    def perform_global_curve_fit(self, data, func, param_values, keep_constant, groups):  
        """
        Perform a non-linear least-squares global fit of func to data 
        @data: list of n_curves curves;
        each curve is an (n_indi + 1, n_points)-shaped array, with n_indi the 
        number of independent ('x') axes and n_points is the number of data
        points in each curve. Curve items [0 : n_indi] contain the values of the 
        independents, and curve item [-1] contains the dependent ('y') values.
        @func: reference to function with signature fn(x, params), with @params  
        a list of parameter values and @x an (n_indi, n_points)-shaped 
        array of independent values.
        @param_values is an (n_curves, n_params)-shaped array with individual values for
        each parameter in each curve.  The values are used as initial estimates 
        (for variable parameters) or as invariants.
        @keep_constant is an (n_curves, n_params)-shaped array of Boolean values 
        (with rows and columns parallel to @curve_names and @param_names, respectively):
        if True, parameter values is an invariant, if False, parameter value is variable.
        @groups is an (n_curves, n_params)-shaped array of integers, in which linked 
        parameters are grouped by their values (the actual value identifying a group
        of linked parameters does not matter, but it has to be unique across all parameters). 
        Example for 4 curves and 3 parameters:
              p0    p1    p2
        c0    0     2     3
        c1    0     2     4
        c2    1     2     5
        c3    1     2     6
        means that parameter p0 is assumed to have the same value in 
        curves c0 and c1, and in curves c2 and c3 (a different value), 
        and that the value for p1 is the same in all curves, whereas
        the value of p2 is different for all curves. In this example, 
        the total number of parameters to be fitted is 7.
        """ 
        # Create a flat data set and an array that indicates where to split 
        # the flat data to reconstruct the individual curves
        process_log = "\n**** New attempt ****\n"
        x_splits = []
        splt = 0
        flat_data = np.array([])
        for curve in data:
            splt += curve.shape[1]
            x_splits.append(splt)
            if flat_data.shape[0] == 0:
                flat_data = curve
            else:
                flat_data = np.concatenate((flat_data, curve), axis=1)
        x_splits = np.array(x_splits[:-1]) # no split at end of last curve
        # Create the input x and y from flat_data
        x, y = flat_data[:-1], flat_data[-1]
        # Get the variables array
        variables = np.logical_not(keep_constant)
        if np.any(variables): # There must be something to fit
            # Get the correct function for global fitting and a first estimate for the variable params
            gfunc, p_est = self.make_func_global(func, x_splits, param_values, variables, groups)
            # Perform the global fit
            pars = None
            ftol, xtol = 1.0e-9, 1.0e-9
            while pars is None and ftol < 1.0:
                try:
                    ftol *= 10.
                    xtol *= 10.
                    out = curve_fit(gfunc, x, y, p0=p_est, ftol=ftol, xtol=xtol, maxfev=250, full_output=1) 
                    pars = out[0]
                    #covar = out[1]
                    nfev = out[2]['nfev']
                    log_entry = "\nNumber of evaluations: " + '{:d}'.format(nfev) + "\tTolerance: " + '{:.1e}'.format(ftol)
                    print(log_entry)
                except ValueError as e:
                    log_entry = "\nValue Error (ass):" + str(e)
                    process_log += log_entry
                    print(log_entry)
                except RuntimeError as e:
                    log_entry = "\nRuntime Error (ass):" + str(e)
                    process_log += log_entry
                    print(log_entry)
                except:
                    log_entry = "\nOther error (ass)"
                    process_log += log_entry
                    print(log_entry)
        
        # Reconstruct and return the full parameter matrix 
        pshp = param_values.shape
        ug, first_occurrence, inverse_indices = np.unique(groups.flatten(), return_index= True, return_inverse=True)
        uv_filter = variables.flatten()[first_occurrence]
        fitted_params = param_values.flatten()[first_occurrence]
        fitted_params[uv_filter] = pars
        param_matrix = fitted_params[inverse_indices].reshape(pshp)
        print(param_matrix)
        return param_matrix
                
        
################################################################################################
## Tests
################################################################################################

def test_global():
    # NOTE: not adapted to latest version of make_func_global, so does not work
    # Keep for example of function with 2 independent axes
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
    fit_y, fit_p, conf_ints = [], [], []
    fp_in = np.ones((npars,))
    c = {}
    for i in range(len(x)):
        fp0 = np.ones_like(fp_in)
        c={1:5.0}
        m = [not (i in c) for i in range(fp0.shape[0])]
        p_est = np.ones(fp0[m].shape, dtype=float)
        pvar, pcov = curve_fit(FunctionsFramework.make_func(FunctionsFramework, fn=ffn, params=fp_in, const=c), x[i], y[i], p0=p_est)
        cis = FunctionsFramework.confidence_intervals(FunctionsFramework, x[i].shape[1], pvar, pcov, 0.95)
        fit_y.append(ffn(x[i], fp_in))
        fit_p.append(fp_in.copy())
        conf_ints.append(cis)
    print(np.array(fit_p))
    print(np.array(conf_ints))
    afit_y = np.array(fit_y)
        
    # Do a global fit
    c = {}
    l_shape = (n_curves, npars)
    groups = np.zeros(l_shape, dtype=int)
    groups[:,1] += 1
    groups[:,2] += 2 # all 3 params are linked in all curves (here)
#    groups = np.array([0,1,2,1,3,1,4,1]).reshape(l_shape)
    #groups = np.arange(n_curves * fp0.shape[0]).reshape(l_shape) # all params independent
    splits = np.arange(n_points, n_curves * n_points, n_points)
    
    flat_x = x[0]
    for x_i in x[1:]:
        flat_x = np.hstack((flat_x, x_i))
    flat_y = y[0]
    for y_i in y[1:]:
        flat_y = np.concatenate((flat_y, y_i))
    ups = np.unique(groups)
    c = {1:5.}
    m = [not (i in c) for i in ups]
    p0 = np.ones_like(ups, dtype=float)[m]
    out = curve_fit(FunctionsFramework.make_func_global(FunctionsFramework, ffn, splits, groups, keep_constant=c), flat_x, flat_y, p0=p0)
    cis = FunctionsFramework.confidence_intervals(FunctionsFramework, flat_y.shape[0], out[0], out[1], 0.95)
    p_fin = np.zeros_like(ups, dtype=float)
    p_fin[m] = out[0]
    for i in c:
        p_fin[i] = c[i]
    print(p_fin)
    p_shape = groups.shape
    flinks = groups.flatten()
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
   

def test_func():   
    import matplotlib.pyplot as plt
    
    x = np.arange(0, 1, 0.001)
    x = x.reshape((1, x.shape[0]))
    f = fdefs.fn_3exp
    p = np.array([1.0, 0.5, 10.0, 0.25, 1.0, 0.0, 1.0])
    c = {5: 0.0, 6: 1.0}
    m = [not (i in c) for i in range(p.shape[0])] # mask only showing variable params
    n_free_params = len(p[m])
    p_est = np.ones(n_free_params, dtype=float)
    data = f(x, p)
    n_points = data.shape[1]
    ndata = data + np.random.normal(0.0, 0.02*data[0], n_points)
    fnc = FunctionsFramework.make_func(FunctionsFramework, f, p, c)
    
    pvar, pcov = curve_fit(fnc, x[0], ndata[0], p0=p_est)
    
    cis = FunctionsFramework.confidence_intervals(FunctionsFramework, n_points, pvar, pcov, 0.95)
    
    plt.plot(x[0], ndata[0], 'ro', x[0], f(x, p)[0], 'k-')
    plt.show()

if __name__ == '__main__':
    test_global()






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