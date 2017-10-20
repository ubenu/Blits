'''
Created on 23 May 2017

@author: schilsm
'''
# -*- coding: utf-8 -*-

import pandas as pd, numpy as np

pd.options.mode.chained_assignment = None  
# suppresses unnecessary warning when creating self.working_data

class BlitsData():
    
    def __init__(self):
        # attributes
        self.file_name = ""
        self.raw_data = None
        self.working_data = None
        self.series_names = None # same as self.series_dict.keys, but in order of input
        # default settings
        self.data_reduction_factor = 5
        
        # New, more general working data
        self.series_dict = {}
     
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
        n_cols = len(self.raw_data.columns)
        named_cols = self.raw_data.columns[~self.raw_data.columns.str.contains('unnamed', case=False)]
        self.series_names = named_cols
        n_series = len(named_cols)
        n_cols_per_series = n_cols // n_series
        assert n_cols % n_series == 0, "Cannot read input"
        n_independents = n_cols_per_series - 1
        # Split data set in individual series
        self.series_dict = {}
        for s in range(0, n_cols , n_cols_per_series):
            df = pd.DataFrame(self.raw_data.iloc[:, s:s+n_cols_per_series]).dropna()
            s_name = df.columns.tolist()[0]
            cn = ['y']
            for i in range(n_independents-1, -1, -1):
                x = 'x{}'.format(i)
                cn.insert(0, x)
            df.columns = cn
            df = df.sort_values(by='x0')
            ix = pd.Index(np.arange(len(df)))
            df.set_index(ix, inplace=True)
            self.series_dict[s_name] = df
            
#         y = self.raw_data.columns[n_cols_per_series-1::n_cols_per_series]
            
#         self.series_names = named_cols #self.raw_data.columns[0::2]         
#         self.working_data = self.raw_data[y]
#         self.working_data.columns = self.series_names
# #        print(self.working_data)
#         self.working_data['time'] = self.raw_data.iloc[:,0]
#         self.working_data = self.working_data[0:-1:self.data_reduction_factor]
        
#     def get_data_x(self):
#         try:
#             return self.working_data['time']
#         except:
#             print("No independent")
#             
#     def get_data_y(self, series_ids=[]):
#         if series_ids == []:
#             series_ids = self.series_names
#         try:
#             return self.working_data[series_ids]  
#         except:
#             print("No dependent")
            

        
 
