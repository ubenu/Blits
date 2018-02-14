'''
Created on 23 May 2017

@author: schilsm
'''
# -*- coding: utf-8 -*-

import pandas as pd, numpy as np, copy as cp

pd.options.mode.chained_assignment = None  
# suppresses unnecessary warning when creating self.working_data

class BlitsData():
    
    def __init__(self):
        # attributes
        self.file_name = ""
        self.raw_data = None
        self.working_data = None
        self.series_names = None # same as self.series_dict.keys, but in order of input
        self.independent_names = None
        # default settings
        self.data_reduction_factor = 5
        
        # New, more general working data
        self.series_dict = {}
        
    def has_data(self):
        return len(self.series_dict) > 0
     
    def import_data(self, file_path):
        self.raw_data = pd.read_csv(file_path)
        self.create_working_data()

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
            
    def create_working_data(self):
        n_cols = len(self.raw_data.columns)
        named_cols = self.raw_data.columns[~self.raw_data.columns.str.contains('unnamed', case=False)]
        self.series_names = named_cols
        n_series = len(named_cols)
        n_cols_per_series = n_cols // n_series
        n_independents = n_cols_per_series - 1
        # Split data set in individual series
        self.series_dict = {}
        self.independent_names = []
        try:
            for s in range(0, n_cols , n_cols_per_series):
                df = pd.DataFrame(self.raw_data.iloc[:, s:s+n_cols_per_series]).dropna()
                s_name = df.columns.tolist()[0]
                self.independent_names = ['x{}'.format(i) for i in range(n_independents)]
                cols = cp.deepcopy(self.independent_names)
                cols.append(s_name)
                df.columns = cols
                df = df.sort_values(by='x0')
                ix = pd.Index(np.arange(len(df)))
                df.set_index(ix, inplace=True)
                self.series_dict[s_name] = df
        except Exception as e:
            print(e)
            
    def create_working_data_from_template(self, template):
        """
        @template:     
        template for series construction, consisting of two pandas DataFrames, 
        with template[0] containing the series axes values and a column for the calculated dependent,
        template[1] containing the parameter values for each axis, and
        template[2] the modelling function        
        """
        n_axes = len(template[2].independents)
        splits = np.arange(1, len(template[0].columns)//(n_axes+1)) * (n_axes+1)
        all_series = np.split(template[0], splits, axis=1)
        self.series_names = []
        self.independent_names = []
        for s in all_series:
            name = s.columns[-1]
            self.series_names.append(name)
            axes_names = s.columns[:-1]
            self.independent_names = cp.deepcopy(axes_names).tolist()  # not pretty: overwrites previous; no check is made
            s_new = cp.deepcopy(s).dropna()
            self.series_dict[name] = s_new
        self.series_names = np.array(self.series_names)
            

            

        
 
