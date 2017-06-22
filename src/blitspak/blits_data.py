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

pd.options.mode.chained_assignment = None  
# suppresses unnecessary warning when creating self.working_data

class BlitsData():
    
    def __init__(self):
        # attributes
        self.file_name = ""
        self.raw_data = None
        self.working_data = None
        self.trace_ids = None
        # default settings
        self.data_reduction_factor = 5
     
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
            
    def _create_working_data(self):
        selected = self.raw_data.columns[1::2]
        self.trace_ids = self.raw_data.columns[0::2]         
        self.working_data = self.raw_data[selected]
        self.working_data.columns = self.trace_ids
        self.working_data['time'] = self.raw_data.iloc[:,0]
        self.working_data = self.working_data[0:-1:self.data_reduction_factor]
        
#     def get_selection(self, start, stop):
#         indmin, indmax = self._get_span_indices(start, stop)
#         return self.working_data[indmin:indmax]
#         
#     def _get_span_indices(self, start, stop):
#         return np.searchsorted(self.working_data['time'],(start, stop))
                
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
            

        
 
