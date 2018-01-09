'''
Created on 24 May 2017

@author: SchilsM
'''

import sys
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import distributions # t
import functions.function_defs as fdefs
#from statsmodels.genmod.tests.results.elastic_net_generate_tests import lin_pred
#from Cython.Debugger.libpython import BuiltInMethodProxy #?? how did that get here?
#from astropy.modeling.tests.test_projections import pars
#from functions import function_defs
#from statsmodels.nonparametric.kernels import d_gaussian

# class ModellingFunction():
#     
#     def __init__(self, name, fn_ref, p0_fn_ref, param_names, fn_str):
#         self.name = name
# 
#         self.fn_ref = fn_ref
#         self.p0_fn_ref = p0_fn_ref
#         self.param_names = param_names
#         self.fn_str = fn_str
#     
#     def __str__(self):
#         return self.fn_str
#     
# class FunctionLibrary():
#     
#     def __init__(self):
#         pass
#     
#     def add_function(self, fn):
#         pass
#     
#     def remove_function(self, fn_id):
#         pass
#     
    
    
class LibraryFunction():
    
    def __init__(self, fn_id, 
                 name='', 
                 short_description='', 
                 long_description='', 
                 fn_def=None, 
                 find_root=None,
                 obs_dependent_name='', 
                 calc_dependent_name='', 
                 independent_names='', 
                 parameter_names='', 
                 first_estimates=None
                 ):
        self.fn_id = fn_id
        self.name = name
        self.short_description = short_description
        self.long_description = long_description
        self.fn_def = fn_def
        self.find_root = find_root,
        self.obs_dependent_name = obs_dependent_name
        self.calc_dependent_name = calc_dependent_name
        if not independent_names is None:
            self.set_independents(independent_names)
        if not parameter_names is None:
            self.set_parameters(parameter_names)
        self.first_estimates = first_estimates
        self.independent_strings = []
        self.parameter_strings = []
        
    def set_independents(self, independent_names=''):
        if independent_names != '':
            self.independent_names = [x.strip() for x in independent_names.split(sep=',')]
            self.independents_df = pd.DataFrame(columns=self.independent_names)
                         
    def set_parameters(self, parameter_names=''):
        self.parameter_strings = [p.strip() for p in parameter_names.split(sep=',')]
        self.parameter_values = np.empty_like(self.parameter_strings, dtype=float)
        
    def test_fn(self):
        print(self.fn_def)
        
    def __str__(self):
        return self.fn_def
    

class FunctionsFramework():     
                  
    def __init__(self):
        pass
    
    def read_modelling_functions(self, file_path):
        self.modfunc_data = pd.read_csv(file_path)
        self.modfunc_data.dropna(inplace=True)
        self.modfunc_data['Id'] = np.nan
        
        # names = self.modfunc_data.loc[self.modfunc_data['Attribute']=='@Name']
        fn_id = 0
        ids = []
        for row in self.modfunc_data.itertuples():
            if row.Attribute == '@Name':
                fn_id += 1
            ids.append(fn_id)
        self.modfunc_data.Id = ids
        unique_ids = np.unique(np.array(self.modfunc_data.Id.tolist()), return_index=True, return_inverse=True, return_counts=True)
        for i in unique_ids[0]: # the actual unique fn_ids
            info = self.modfunc_data.loc[self.modfunc_data['Id']==i]
            name = info.loc[info['Attribute'] == '@Name']['Value'].values[0]
            sd = info.loc[info['Attribute'] == '@Short description']['Value'].values[0]
            ld = info.loc[info['Attribute'] == '@Long description']['Value'].values[0]
            fn = info.loc[info['Attribute'] == '@Function']['Value'].values[0]
            rt = info.loc[info['Attribute'] == '@FindRoot']['Value']
            odp = info.loc[info['Attribute'] == '@Observed dependent']['Value'].values[0]
            cdp = info.loc[info['Attribute'] == '@Calculated dependent']['Value'].values[0]
            idp = info.loc[info['Attribute'] == '@Independents']['Value'].values[0]
            par = info.loc[info['Attribute'] == '@Parameters']['Value'].values[0]
            est = info.loc[info['Attribute'] == '@First estimates']['Value'].values[0]

            lib_func = LibraryFunction(i)
            lib_func.name = name
            lib_func.short_description = sd
            lib_func.long_description = ld
            lib_func.fn_def = fn
            if len(rt):
                lib_func.find_root = rt
            lib_func.obs_dependent_name = odp
            lib_func.calc_dependent_name = cdp
            lib_func.set_independents(idp) 
            lib_func.set_parameters(par)
            lib_func.first_estimates = est
                                
#             lib_func.test_fn()
            
                
            
            
        
    def confidence_intervals(self, n, params, covar, conf_level):
        """
        @n is the number of data points used for the estimation of params and covar
        @params is a 1D numpy array of best-fit parameter values
        @covar is the best fit covariance matrix
        @conf_level is the required confidence level, eg 0.95 for 95% confidence intervals
        @return a 1D numpy array of the size of params with the confidence intervals
        on params (report (eg) p +/- d or p (d/p*100 %), where p is a parameter value
        and d is its associated relative confidence interval.
        """
        dof = max(0, n - params.shape[0]) # degrees of freedom
        tval = distributions.t.ppf(conf_level / 2., dof) # student-t value for the dof and confidence level
        sigma = self.standard_error_from_covar(covar)
        return sigma * tval
    
    def standard_error_from_covar(self, covar):
        """
        @covar is a covariance matrix
        @return a 1D numpy array representing the standard error on the data, derived from
        the covariance matrix
        """
        return np.power(np.diag(covar), 0.5) # standard error
        
            
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
        
        @return [0] a function that can be used as input to curve_fit
        @return [1] a flat array with the unique variable parameter values
        
        """
        
        # ugroups are the IDs of the unique groups in a flattened array
        # indices are the indices of the first occurrence of a unique group 
        #    in the flattened array or in a parallel one
        # reverse_indices indicates where to find a particular ID in ugroups 
        #    to reconstruct the original flat array (or a parallel one)
        ugroups, indices, inverse_indices = np.unique(groups.flatten(), 
                                                      return_index=True, 
                                                      return_inverse=True)
        pshape = param_vals.shape
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
                          
    def perform_global_curve_fit(self, data, func, param_values, keep_constant, groups):  
        """
        Perform a non-linear least-squares global fit of func to data 
        @data: list of n_curves curves;
        each curve is an (n_indi + 1, n_points)-shaped numpy array, with n_indi the 
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
        
        @return [0] (numpy array) parameter_matrix: full parameter value matrix, shape as @param_values
        @return [1] (numpy array) confidence_matrix: full confidence intervals matrix, shape as @param_values 
                (could be reported to user as, eg, confidence_matrix/parameter_matrix for 
                estimate of stdev on returned parameter values
        @return [2] (float) the ftol value achieved (the smaller the better, starts at 1e-8, maximally 1e0)
        @return [3] (str) the process log, with details of the fitting process
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
        ftol, xtol = 0., 0.
        variables = np.logical_not(keep_constant)
        gfunc, p_est = self.make_func_global(func, x_splits, param_values, variables, groups)
        pars, sigmas, conf_ints = None, None, None
        parameter_matrix, sigma_matrix, confidence_matrix = None, None, None
        if np.any(variables): # There is something to fit
            # Get the correct function for global fitting and a first estimate for the variable params
            # gfunc, p_est = self.make_func_global(func, x_splits, param_values, variables, groups)
            # Perform the global fit
            # pars, sigmas, conf_ints = None, None, None
            # parameter_matrix, sigma_matrix, confidence_matrix = None, None, None
            ftol, xtol = 1.0e-9, 1.0e-9
            while ftol < 1.0 and pars is None:
                try:
                    ftol *= 10.
                    xtol *= 10.
                    out = curve_fit(gfunc, x, y, p0=p_est, ftol=ftol, xtol=xtol, maxfev=250, full_output=1) 
                    pars = out[0] 
                    covar = out[1]
                    sigmas = self.standard_error_from_covar(covar)
                    conf_ints = self.confidence_intervals(x.shape[1], pars, covar, 0.95) 
                    nfev = out[2]['nfev']
                    log_entry = "\nNumber of evaluations: " + '{:d}'.format(nfev) + "\tTolerance: " + '{:.1e}'.format(ftol)
                    process_log += log_entry
                except ValueError as e:
                    log_entry = "\nValue Error (ass):" + str(e)
                    process_log += log_entry
                except RuntimeError as e:
                    log_entry = "\nRuntime Error (ass):" + str(e)
                    process_log += log_entry
                except:
                    log_entry = "\nOther error (ass)"
                    process_log += log_entry
            
        # Reconstruct and return the full parameter matrix 
        if not pars is None:
            ug, first_occurrence, inverse_indices = np.unique(groups.flatten(), return_index= True, return_inverse=True)
            uv_filter = variables.flatten()[first_occurrence]
            fitted_params = param_values.flatten()[first_occurrence]
            sigma = np.zeros_like(fitted_params)
            confidence = np.zeros_like(fitted_params)
            fitted_params[uv_filter] = pars
            sigma[uv_filter] = sigmas  
            confidence[uv_filter] = conf_ints
            parameter_matrix = fitted_params[inverse_indices].reshape(param_values.shape)
            sigma_matrix = sigma[inverse_indices].reshape(param_values.shape)
            confidence_matrix = confidence[inverse_indices].reshape(param_values.shape)
        else: # There was nothing to fit; all parameters are constant
            parameter_matrix = param_values
            sigma_matrix = np.zeros_like(parameter_matrix)
            confidence_matrix = np.zeros_like(parameter_matrix)
            
        return parameter_matrix, sigma_matrix, confidence_matrix, ftol, process_log
                


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


