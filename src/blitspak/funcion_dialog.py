'''
Created on 9 Jan 2018

@author: schilsm
'''

from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets

import pandas as pd, numpy as np


NROWS = 5
FNAME, NINDEPENDENTS, NPARAMS, DESCRIPTION, DEFINITION = range(NROWS)


if __name__ == '__main__':
    pass


class FunctionSelectionDialog(widgets.QDialog):
    
    def __init__(self, parent=None):
        super(FunctionSelectionDialog, self).__init__(parent)
        
        self.model = FunctionLibrary()
        self.setWindowTitle("Modelling function selection")
        
        main_layout = widgets.QVBoxLayout()
        button_box = widgets.QDialogButtonBox()
        button_box.addButton(widgets.QDialogButtonBox.Apply)
        button_box.addButton(widgets.QDialogButtonBox.Abort)
         
        table_label = widgets.QLabel("Available functions")
        self.tableview = widgets.QTableView()
        table_label.setBuddy(self.tableview)
        self.tableview.setModel(self.model)

        main_layout.addWidget(self.tableview)
        main_layout.addWidget(button_box)
        self.setLayout(main_layout)
          
 
class ModellingFunction(qt.QAbstractTableModel):
    
    def __init__(self, id):
        super(ModellingFunction, self).__init__()
        
        self.name = ""
        self.short_description = ""
        self.long_description = ""
        self.fn_def = ""
        self.find_root = ""
        self.obs_dependent_name = ""
        self.calc_dependent_name = ""
        self.independents = "" 
        self.parameters = ""
        self.first_estimates = ""        

 
class FunctionLibrary(qt.QAbstractTableModel):
    
    def __init__(self, filepath="C:\\Users\\schilsm\\git\\Blits\\Resources\\ModellingFunctions\\Functions.csv"):
        super(FunctionLibrary, self).__init__()
        
        self.filepath = filepath
        
        self.dirty = False
        
        self.load_lib()
        
    
    def load_lib(self):    
        self.ships = []
        raw_data = pd.read_csv(self.filepath)
        print (raw_data)

        self.raw_data.dropna(inplace=True)
        self.raw_data['Id'] = np.nan
        
        # names = self.raw_data.loc[self.raw_data['Attribute']=='@Name']
        fn_id = 0
        ids = []
        for row in self.raw_data.itertuples():
            if row.Attribute == '@Name':
                fn_id += 1
            ids.append(fn_id)
        self.raw_data.Id = ids
        unique_ids = np.unique(np.array(self.raw_data.Id.tolist()), return_index=True, return_inverse=True, return_counts=True)
        for i in unique_ids[0]: # the actual unique fn_ids
            info = self.raw_data.loc[self.raw_data['Id']==i]
            name = info.loc[info['Attribute'] == 'Name']['Value'].values[0]
            sd = info.loc[info['Attribute'] == 'Short description']['Value'].values[0]
            ld = info.loc[info['Attribute'] == 'Long description']['Value'].values[0]
            fn = info.loc[info['Attribute'] == 'Function']['Value'].values[0]
            rt = info.loc[info['Attribute'] == 'FindRoot']['Value']
            odp = info.loc[info['Attribute'] == 'Observed dependent']['Value'].values[0]
            cdp = info.loc[info['Attribute'] == 'Calculated dependent']['Value'].values[0]
            idp = info.loc[info['Attribute'] == 'Independents']['Value'].values[0]
            par = info.loc[info['Attribute'] == 'Parameters']['Value'].values[0]
            est = info.loc[info['Attribute'] == 'First estimates']['Value'].values[0]

            lib_func = ModellingFunction(i)
            lib_func.name = name
            lib_func.short_description = sd
            lib_func.long_description = ld
            lib_func.fn_def = fn
            if len(rt):
                lib_func.find_root = rt
            lib_func.obs_dependent_name = odp
            lib_func.calc_dependent_name = cdp
            lib_func.independents = idp 
            lib_func.set_parameters = par
            lib_func.first_estimates = est
        
       