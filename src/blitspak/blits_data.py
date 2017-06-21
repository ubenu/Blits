'''
Created on 23 May 2017

@author: schilsm
'''
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 10:31:43 2016

@author: Maria
"""

import numpy as np
import pandas as pd
#from scipy.optimize import curve_fit

pd.options.mode.chained_assignment = None  
# suppresses unnecessary warning when creating self.working_data
#import warnings
#warnings.simplefilter(action = "ignore", category = RuntimeWarning)
# suppresses unnecessary runtime warnings when fitting data

class BlitsData():
    
    def __init__(self):
        
#         self.functions = {'fn_line': self.fn_straight_line,
#                           'fn_1exp': self.fn_1exp,
#                           'fn_2exp': self.fn_2exp,
#                           'fn_3exp': self.fn_3exp,
#                           'fn_hill': self.fn_hill }

#         # templates for results
#         self.residuals = {'baseline':None, 'loaded': None, 'association': None}
#         self.fitted = {'baseline':None, 'loaded': None, 'association': None}
#         self.assoc_params = None
#         self.results = None
#         self.fractional_saturation_params = None

        # attributes
        self.file_name = ""
        self.raw_data = None
        self.working_data = None
        self.trace_ids = None
#         self.results_acquired = {'baseline': False, 
#                                  'loaded': False, 
#                                  'association': False,
#                                  'fractional saturation': False }
#         self.current_message = ""
        
        # default settings
        self.data_reduction_factor = 5
#         self.fit_func_id = 'fn_2exp'
#         self.frac_sat_func_id = 'fn_hill'
     
    def import_data(self, file_path):
        self.raw_data = pd.read_csv(file_path)
        self._create_working_data()

    def export_results(self, file_path):
        r = self.results.to_csv()
        p = self.get_fractional_saturation_params_dataframe().to_csv()
        f = self.get_fractional_saturation_curve().to_csv()
        with open(file_path, 'w') as file:
            file.write(r)
            file.write('\n')
        with open(file_path, 'a') as file:
            file.write(p)
            file.write('\n')
            file.write(f)
            
    def get_selection(self, start, stop):
        indmin, indmax = self._get_span_indices(start, stop)
        return self.working_data[indmin:indmax]
        
#     def set_baseline_measurements(self, start, stop):
#         if self.results is None:
#             self.results = self._create_results_template(self.trace_ids) 
#         indmin, indmax = self._get_span_indices(start, stop)
#         selection = self.working_data[indmin:indmax]
#         sel_ones = selection.copy()
#         sel_ones[self.trace_ids] = 1.0
#         t = selection['time']
#         means = selection[self.trace_ids].mean()
#         self.results['baseline'] = means
#         self.results_acquired['baseline'] = True
#         self.set_measurements()
#         res = selection[self.trace_ids] - means
#         res['time'] = t
#         fit = sel_ones.copy() * means
#         fit['time'] = t
#         self.residuals['baseline'] = res
#         self.fitted['baseline'] = fit
#             
#     def set_loads_measurements(self, start, stop):
#         if self.results is None:
#             self.results = self._create_results_template(self.trace_ids)            
#         indmin, indmax = self._get_span_indices(start, stop)
#         selection = self.working_data[indmin:indmax]
#         sel_ones = selection.copy()
#         sel_ones[self.trace_ids] = 1.0
#         t = selection['time']
#         means = selection[self.trace_ids].mean()
#         self.results['loaded'] = means
#         self.results_acquired['loaded'] = True
#         self.set_measurements()
#         res = selection[self.trace_ids] - means
#         res['time'] = t
#         fit = sel_ones.copy() * means
#         fit['time'] = t
#         self.residuals['loaded'] = res
#         self.fitted['loaded'] = fit
        

#     def set_association_measurements(self, start, stop):
#         indmin, indmax = self._get_span_indices(start, stop)
#         selection = self.working_data[indmin:indmax]
#         self.results['association'] = 0.0
#         self.results['success'] = 0.0
#         pnames = self.get_parameter_names(self.fit_func_id)
#         self.assoc_params = pd.DataFrame(np.zeros((len(self.trace_ids), 
#                                                              len(pnames)), dtype=float))
#         self.assoc_params.columns = pnames
#         self.assoc_params.index = self.trace_ids
#         res = selection.copy() 
#         res[self.trace_ids] = 0.0
#         fit = res.copy()
#         self.residuals['association'] = res
#         self.fitted['association'] = fit
#         func = self.functions[self.fit_func_id]
#         t = selection['time']
#         x = t - t.iloc[0]  
#         for trace in self.trace_ids:
#             y = selection[trace]
#             p_est = self.get_initial_estimates(self.fit_func_id, x, y)
#             params = None
#             try:
#                 cf_out = curve_fit(func, x, y, p0=p_est)
#                 params = cf_out[0]
#                 self.results['association'][trace] = params[0]
#                 self.results['success'][trace] = 1.0
#                 for i in range(len(pnames)):
#                     self.assoc_params[pnames[i]][trace] = params[i]
#                 p = tuple(params)
#                 y_fit = func(x, *p)
#                 y_res = y - y_fit
#                 self.residuals['association'][trace] = y_res
#                 self.fitted['association'][trace] = y_fit
#             except ValueError as e:
#                 self.current_message = "Value Error (ass):" + str(e)
#             except RuntimeError as e:  
#                 self.current_message = "Runtime Error (ass):" + str(e)
#             except:
#                 self.current_message = "Other error (ass)"
#         if not params is None:
#             self.results_acquired['association'] = True
#             self.set_measurements()
        
#     def set_measurements(self):
#         if self.results_acquired['baseline'] and self.results_acquired['loaded']:
#             sl = (self.results['loaded'] - self.results['baseline'])
#             self.results['Sugar loading'][self.trace_ids] = sl
#             if self.results_acquired['association']:
#                 amp = (self.results['loaded'] - self.results['association'])
#                 self.results['Amplitude (obs)'][self.trace_ids] = amp
#                 self.set_fractional_saturation_results()

#     def set_fractional_saturation_results(self):
#         if (self.results_acquired['baseline'] and
#             self.results_acquired['loaded'] and
#             self.results_acquired['association']):
#             func = self.functions[self.frac_sat_func_id]
#             data = pd.DataFrame()
#             data['Sugar loading'] = self.results['Sugar loading']
#             data['Amplitude (obs)'] = self.results['Amplitude (obs)']
#             mask = self.results['success'] == 1.0
#             params, y_calc = None, None
#             if mask.any():
#                 temp = data[mask]
#                 data = temp
#             data = data.sort_values('Sugar loading')
#             try:
#                 x = data['Sugar loading']
#                 y = data['Amplitude (obs)']
#                 p_est = self.get_initial_estimates(self.frac_sat_func_id, x, y)
#                 cf_out = curve_fit(func, x, y, p0=p_est, method='trf')
#                 params = cf_out[0]
#             except ValueError as e:
#                 print("Value Error (frac sat):" + str(e))
#             except RuntimeError as e:  
#                 print("Runtime Error (frac sat):" + str(e))
#             except Exception as e:
#                 print("Other error (frac sat)" + str(e))
#     
#             if not params is None:
#                 p = tuple(params)
#                 self.fractional_saturation_params = {'ymax': p[0],'xhalf': p[1],
#                                                      'h': p[2]}
#                 x = self.results['Sugar loading']
#                 y_calc = func(x, *p)
#                 self.results['Amplitude (calc)'][self.trace_ids] = y_calc
#                 self.results_acquired['fractional saturation'] = True

#     def get_fractional_saturation_params_dataframe(self):
#         p = self.fractional_saturation_params
#         df = pd.DataFrame(np.zeros((1,3)))
#         df.index = [self.file_name]
#         df.columns = ['y-max', 'x-half-sat', 'h']
#         df['y-max'] = [p['ymax']]
#         df['x-half-sat'] = [p['xhalf']]
#         df['h'] = [p['h']]
#         return df
# 
#                     
#     def get_fractional_saturation_curve(self, size=3.5, step=0.005):
#         max_x = max(size * self.fractional_saturation_params['xhalf'], 
#                     self.results['Sugar loading'].max())
#         data = pd.DataFrame()
#         func = self.functions[self.frac_sat_func_id]
#         ym = self.fractional_saturation_params['ymax']
#         xh = self.fractional_saturation_params['xhalf']
#         h = self.fractional_saturation_params['h']
#         p = (ym, xh, h)
#         data['x'] = np.arange(0.0, max_x, step) 
#         y = func(data['x'], *p)
#         data['y'] = y
#         return data 

#     def get_fractional_saturation_residuals(self, params):
#         x = self.results['Sugar loading']
#         y_obs = self.results['Amplitude (obs)']
#         func = self.functions[self.frac_sat_func_id]
#         p = tuple(params)
#         y_calc = func(x, *p)
#         return y_obs - y_calc
#             
#     def _create_results_template(self, trace_ids):
#         header = ['baseline', 'loaded', 'association', 'success',
#                   'Sugar loading', 'Amplitude (obs)', 'Amplitude (calc)']
#         data = np.zeros((len(trace_ids), len(header)), dtype=float)
#         template = pd.DataFrame(data)
#         template.index = trace_ids
#         template.columns = header
#         return template
#         
    def _create_working_data(self):
        selected = self.raw_data.columns[1::2]
        self.trace_ids = self.raw_data.columns[0::2]         
        self.working_data = self.raw_data[selected]
        self.working_data.columns = self.trace_ids
        self.working_data['time'] = self.raw_data.iloc[:,0]
        self.working_data = self.working_data[0:-1:self.data_reduction_factor]
        
    def _get_span_indices(self, start, stop):
        return np.searchsorted(self.working_data['time'],(start, stop))
                        
#     def get_initial_estimates(self, func_id, x, y):
#         if func_id in ['fn_line', 'fn_1exp', 'fn_2exp', 'fn_3exp']:
#             n = {'fn_line': 2, 'fn_1exp': 3, 'fn_2exp': 5, 'fn_3exp': 7}
#             a0 = y.min()
#             a1 = (y.min() - y.max()) / 2.0
#             k1 = 4.0 / x.max()
#             a2 = a1 / 2.0
#             k2 = 2.0 / x.max()
#             a3 = a2 / 2.0
#             k3 = 1.0 / x.max()
#             estimates = (a0, a1, k1, a2, k2, a3, k3)
#             return estimates[0 : n[func_id]]
#         elif func_id == 'fn_hill':
#             ymax, xhalf, h = y.max(), x.max(), 1.0
#             return(ymax, xhalf, h)
#         
#     def get_parameter_names(self, func_id):
#         if func_id == 'fn_line':
#             return['a', 'b']
#         elif func_id == 'fn_1exp':
#             return ['a0', 'a1', 'k1']
#         elif func_id == 'fn_2exp':
#             return ['a0', 'a1', 'k1', 'a2', 'k2']
#         elif func_id == 'fn_3exp':
#             return ['a0', 'a1', 'k1', 'a2', 'k2', 'a3', 'k3']
#         elif func_id == 'fn_hill':
#             return ['ymax', 'xhalf', 'h']
#         else:
#             return None        
                
    def get_data_x(self):
        try:
            return self.working_data['time']
        except:
            print("No independent")
            
    def get_data_y(self, trace_ids=[]):
        if trace_ids == []:
            trace_ids = self.trace_ids
        try:
            return self.working_data[trace_ids]  
        except:
            print("No dependent")
            
#     def fn_straight_line(self, x, *p):
#         a, b = p
#         return (a + b*x)
#     
#     def fn_1exp(self, x, *p):
#         a0, a1, k1 = p
#         return (a0 + a1*np.exp(-x*k1))
#         
#     def fn_2exp(self, x, *p):
#         a0, a1, k1, a2, k2 = p
#         return (a0 + a1*np.exp(-x*k1) + a2*np.exp(-x*k2))
#         
#     def fn_3exp(self, x, *p):
#         a0, a1, k1, a2, k2, a3, k3 = p
#         return (a0 + a1*np.exp(-x*k1) + a2*np.exp(-x*k2) + a3*np.exp(-x*k3))
#         
#     def fn_hill(self, x, *p):
#         ymax, xhalf, h = p
#         return ymax / (np.power(xhalf/x, h) + 1.0)
        
 
