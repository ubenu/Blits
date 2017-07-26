'''
Created on 24 May 2017

@author: SchilsM
'''

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import distributions # t
import functions.function_defs as fdefs
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
            
    def make_func(self, fn, params, const={}):
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
    
    
    def make_func_global(self, fn, x_splits, links, consts={}):
        """
        @fn is a function with signature fn(x, p), where @p is 
        a 1D array of parameters and @x is a (k, m) shaped array, 
        with k the number of independents and m the total number of 
        data points (all curves concatenated along the array's 2nd axis)
        @x_splits is  is a sorted 1-D array of integers, whose entries 
        indicate where the array's second axis must be split to generate
        the individual curves.
        @links is an array of shape (n_curves, n_params) of integers, 
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
        @consts is a dictionary with the indices (keys) and values (values) 
        of the parameters that must be kept constant during the fit. 
        """
        n_curves = links.shape[0]
        n_params = links.shape[1]
        links = links.flatten() 
        def func(x, *v):
            split_x = np.split(x, x_splits, axis=1)
            params = np.zeros_like(links, dtype=float)
            u = np.zeros_like(np.unique(links), dtype=float) 
            mask = np.ones_like(u, dtype=bool)
            for i in consts:
                u[i] = consts[i]
                mask[i] = False
            # u will contain the values of all parameters
            u[mask] = v
            # input array v contains the values of the variable parameters
            for i in range(links.shape[0]):
                params[i] = u[links[i]] # fill in the parameter values
            params = params.reshape((n_curves, n_params))
            y_out = []
            for x, p in zip(split_x, params):
                i_out = fn(x, p)
                y_out.extend(i_out)
            return y_out      
        return func 
    
    def perform_global_curve_fit(self, curve_order, param_order, 
                                 data, func, param_values, consts, links):  
        """
        Perform a non-linear least-squares global fit of func to data 
        @curve_order is an array of curve keys in the order in which the curves
        should be concatenated (note to self: is this necessary?)
        @param_order is an array of parameter identifiers in the order in which
        the parameters appear in func
        @data is a dictionary of the nc individual curves to which 
        the global fit is to be applied, with the curve identifiers
        formind the keys. The values are the (x,y) data for the curves, which
        must be (nx+1, mc)-shaped arrays with nx the number of independents
        (number of 'x-axes') and mc the total number of data points in
        that particular curve. The number of independents, nx, must
        be the same in each curve.  The first nx rows must contain the 
        independent values, and the bottom row (data[i][-1]) must contain the 
        dependent values ('y-values').
        @func is a function with signature fn(x, p), where @p is 
        a 1D array of parameters and @x is a (k, m) shaped array, 
        with k the number of independents and m the total number of 
        data points. 
        @param_values is a dictionary  whose keys are curve identifiers
        and whose values are dictionaries with the parameter identifiers
        as keys and the initial or constant value for that parameter as values.  
        @consts is a dictionary of curve identifiers (keys) with a list of
        parameter ids to be kept constant (at the value in @param_values)
        for each curve identifier.
        @links is an array of shape (n_curves, n_params) of integers, 
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
        """ 
        # Prepare the input for make_func_global and curve_fit
        # func and links are already in order
        # data needs to be concatenated and the x_splits array constructed
        x_splits = []
        splt = 0
        flat_data = None
        for ckey in curve_order:
            splt += data[ckey].shape[1]
            x_splits.append(splt)
            if flat_data is None:
                flat_data = data[ckey]
            else:
                flat_data = np.concatenate((flat_data, data[ckey]), axis=1)
        x_splits = np.array(x_splits[:-1]) # no split at end of last curve
        # param_values needs to be put in a 2D array
        all_params = np.empty((len(curve_order), len(param_order)))
        for key in param_values:
            ic = curve_order.index(key)
            all_params[ic] = param_values[key]

        uniq_params, first_occurrence, inverse_indices = np.unique(links, return_index= True, return_inverse=True)
                
        # constants must be translated into the required format, 
        # and we also need the variable parameter estimates in the correct format
        const_values = {}
        const_params_mask = np.zeros_like(all_params, dtype=bool)
        var_params_mask = np.ones_like(all_params, dtype=bool)       
        for key in consts:
            ic = curve_order.index(key)
            for p in consts[key]:
                ip = param_order.index(p)
                nparam = links[ic][ip] # gives the param nr in the links matrix
                vparam = all_params[ic][ip] # gives the value in the param values matrix
                var_params_mask[ic][ip] = False
                const_params_mask[ic][ip] = True
                const_values[nparam] = vparam # so this will overwrite any earlier value
                # To avoid confusion, should deal with this before it gets here (ie in UI)
        # we also need parameter estimates in the correct format (with the constants excluded)
        uniq_param_values = all_params.flatten()[first_occurrence]
        uniq_var_params_mask = var_params_mask.flatten()[first_occurrence]
        uniq_const_params_mask = const_params_mask.flatten()[first_occurrence]
        p_est = uniq_param_values[uniq_var_params_mask]
        x = np.reshape(flat_data[0], (1, flat_data[0].shape[0])) ## Note: this is only for 1D independent - need to check for higher dimensions
        y = flat_data[-1]
        gfunc = self.make_func_global(self, func, x_splits, links, consts=const_values)
        pars, covs = curve_fit(gfunc, x, y, p0=p_est)
        uniq_param_values[uniq_var_params_mask] = pars
        all_params_reconstructed = uniq_param_values[inverse_indices].reshape(all_params.shape)
        
        
        


        

    
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
    out = curve_fit(FunctionsFramework.make_func_global(FunctionsFramework, ffn, splits, links, consts=c), flat_x, flat_y, p0=p0)
    cis = FunctionsFramework.confidence_intervals(FunctionsFramework, flat_y.shape[0], out[0], out[1], 0.95)
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
    npoints = data.shape[1]
    ndata = data + np.random.normal(0.0, 0.02*data[0], npoints)
    fnc = FunctionsFramework.make_func(FunctionsFramework, f, p, c)
    
    pvar, pcov = curve_fit(fnc, x[0], ndata[0], p0=p_est)
    
    cis = FunctionsFramework.confidence_intervals(FunctionsFramework, npoints, pvar, pcov, 0.95)
    
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