'''
Created on 9 Jan 2018

@author: schilsm
'''

from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets

import pandas as pd, numpy as np, copy as cp

import functions.function_defs as fdefs


if __name__ == '__main__':
    pass

class FunctionSelectionDialog(widgets.QDialog):
    
    def __init__(self, parent, n_axes=np.inf, selected_fn_name=""):
        super(FunctionSelectionDialog, self).__init__(parent)
        self.setModal(False)
                
        self.setWindowTitle("Select modelling function")
        main_layout = widgets.QVBoxLayout()
        self.button_box = widgets.QDialogButtonBox()
        self.button_box.addButton(widgets.QDialogButtonBox.Cancel)
        self.button_box.addButton(widgets.QDialogButtonBox.Ok)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        table_label = widgets.QLabel("Available functions")
        self.tableview = widgets.QTableView()
        table_label.setBuddy(self.tableview)
        self.model = FunctionLibraryTableModel(n_axes=n_axes)
        self.tableview.setModel(self.model)
        self.tableview.setSelectionBehavior(widgets.QAbstractItemView.SelectRows)
        self.tableview.setSelectionMode(widgets.QAbstractItemView.SingleSelection)
        self.tableview.doubleClicked.connect(self.accept)
        
        self.selected_fn_name = selected_fn_name
        if self.selected_fn_name != "":
            self.tableview.selectRow(self.findItem(self.selected_fn_name).row())
                
        main_layout.addWidget(self.tableview)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)
        
    def findItem(self, item_text):
        proxy = qt.QSortFilterProxyModel()
        proxy.setSourceModel(self.model)
        proxy.setFilterFixedString(item_text)
        matching_index = proxy.mapToSource(proxy.index(0,0))
        return matching_index
    
    def set_selected_function_name(self):
        row, col = self.tableview.selectionModel().currentIndex().row(), self.model.FNAME
        i = self.tableview.model().index(row, col, qt.QModelIndex())
        self.selected_fn_name = self.tableview.model().data(i).value()
     
    def get_selected_function(self):
        if self.selected_fn_name in self.model.funcion_dictionary:
            return self.model.funcion_dictionary[self.selected_fn_name]
        return None
    
    def accept(self):
        self.set_selected_function_name()
        widgets.QDialog.accept(self)
        
    def reject(self):
        widgets.QDialog.reject(self)
          
 
class ModellingFunction(object):
    
    def __init__(self, uid):
        """
        @uid: unique identifier (int)
        """
        super(ModellingFunction, self).__init__()
        
        self.uid = uid
        self.name = ""
        self.description = ""
        self.long_description = ""
        self.definition = ""
        self.find_root = ""
        self.obs_dependent_name = ""
        self.calc_dependent_name = ""
        self.independents = "" 
        self.parameters = None
        self.first_estimates = ""
        self.func = None 
        self.p0 = None  
        
    def get_parameter_names(self):
        if self.parameters is not None:
            return cp.deepcopy(self.parameters)  
        return None 
    
    def get_axes_names(self):
        if self.independents is not None:
            return cp.deepcopy(self.independents)
        return None
    
    def get_uid(self):
        return cp.deepcopy(self.uid)  

    def get_description(self):
        return self.description  

    def get_long_description(self):
        return self.long_description  

 
class FunctionLibraryTableModel(qt.QAbstractTableModel):

    NCOLS = 5
    FNAME, INDEPENDENTS, PARAMS, DESCRIPTION, DEFINITION = range(NCOLS)
    M_FUNC, M_P0 = range(2)
    fn_dictionary = {
        "Mean": (
            fdefs.fn_average,
            fdefs.p0_fn_average,
            ),
        "Straight line": (
            fdefs.fn_straight_line,
            fdefs.p0_fn_straight_line,
            ),
        "Single exponential decay": (
            fdefs.fn_1exp, 
            fdefs.p0_fn_1exp,
            ),
        "Single exponential decay and straight line": (
            fdefs.fn_1exp_strline, 
            fdefs.p0_fn_1exp_strline,
            ),
        "Double exponential decay": (
            fdefs.fn_2exp,
            fdefs.p0_fn_2exp,
            ),
        "Double exponential and straight line": (
            fdefs.fn_2exp_strline,
            fdefs.p0_fn_2exp_strline,
            ), 
        "Triple exponential decay": (
            fdefs.fn_3exp, 
            fdefs.p0_fn_3exp,
            ),
        "Michaelis-Menten model (initial rates)": (
            fdefs.fn_mich_ment,
            fdefs.p0_fn_mich_ment,
            ),
        "Competitive enzyme inhibition model (initial rates)": (
            fdefs.fn_comp_inhibition,
            fdefs.p0_fn_comp_inhibition,
            ), 
        "Uncompetitive enzyme inhibition": (
            fdefs.fn_uncomp_inhibition,
            fdefs.p0_fn_uncomp_inhibition,
            ),
        "Noncompetitive enzyme inhibition": (
            fdefs.fn_noncomp_inhibition,
            fdefs.p0_fn_noncomp_inhibition,
            ),
        "Mixed enzyme inhibition": (
            fdefs.fn_mixed_inhibition,
            fdefs.p0_fn_mixed_inhibition,
            ),
        "Hill equation": (
            fdefs.fn_hill,
            fdefs.p0_fn_hill,
            ),
        "Competitive 2-ligand binding (Absorbance/Fluorescence)": (
            fdefs.fn_comp_binding,
            fdefs.p0_fn_comp_binding,
            ),
        "Chemical denaturation": (
            fdefs.fn_chem_unfold,
            fdefs.p0_fn_chem_unfold,
            ),
        "Thermal denaturation": (
            fdefs.fn_therm_unfold,
            fdefs.p0_fn_therm_unfold,
            ),
        }
        
    def __init__(self, n_axes, filepath="..\\..\\Resources\\ModellingFunctions\\Functions.csv"):
        super(FunctionLibraryTableModel, self).__init__()
        
        self.filepath = filepath
        self.raw_data = None
        self.dirty = False
        
        self.modfuncs = []
        self.funcion_dictionary = {}
        self.load_lib(n_axes)        
    
    def load_lib(self, n_axes):    
        self.modfuncs = []
        self.funcion_dictionary = {}
        
        self.raw_data = pd.read_csv(self.filepath)
        self.raw_data.dropna(inplace=True)
        self.raw_data['uid'] = np.nan
        
        fn_id = 0
        ids = []
        for row in self.raw_data.itertuples():
            if row.Attribute == 'Name':
                fn_id += 1
            ids.append(fn_id)
        self.raw_data.uid = ids
        unique_ids = np.unique(np.array(self.raw_data.uid.tolist()), return_index=True, return_inverse=True, return_counts=True)
        
        for i in unique_ids[0]: # the actual unique fn_ids
            info = self.raw_data.loc[self.raw_data['uid']==i]
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

            modfunc = ModellingFunction(i)
            modfunc.name = name
            modfunc.description = sd
            modfunc.long_description = ld
            modfunc.definition = fn
            if len(rt):
                modfunc.find_root = rt
            modfunc.obs_dependent_name = odp.strip()
            modfunc.calc_dependent_name = cdp.strip()
            modfunc.independents = [i.strip() for i in idp.split(',')]
            if '' in modfunc.independents:
                modfunc.independents.remove('')
            modfunc.parameters = [i.strip() for i in par.split(',')]
            if '' in modfunc.parameters:
                modfunc.parameters.remove('')
            modfunc.first_estimates = est
            modfunc.func = self.fn_dictionary[modfunc.name][self.M_FUNC]
            modfunc.p0 = self.fn_dictionary[modfunc.name][self.M_P0]
            if len(modfunc.independents) <= n_axes:
                self.modfuncs.append(modfunc)
                self.funcion_dictionary[modfunc.name] = modfunc
            
    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        # Implementation of super.headerData
        if role == qt.Qt.TextAlignmentRole:
            if orientation == qt.Qt.Horizontal:
                return qt.QVariant(int(qt.Qt.AlignLeft|qt.Qt.AlignVCenter))
            return qt.QVariant(int(qt.Qt.AlignRight|qt.Qt.AlignVCenter))
        if role != qt.Qt.DisplayRole:
            return qt.QVariant()
        if orientation == qt.Qt.Horizontal:
            if section == self.FNAME:
                return qt.QVariant("Name")
            elif section == self.INDEPENDENTS:
                return qt.QVariant("Independents")
            elif section == self.PARAMS:
                return qt.QVariant("Parameters")
            elif section == self.DESCRIPTION:
                return qt.QVariant("Description")
            elif section == self.DEFINITION:
                return qt.QVariant("Definition")
        return qt.QVariant(int(section + 1))

    def rowCount(self, index=qt.QModelIndex()):
        return len(self.modfuncs)

    def columnCount(self, index=qt.QModelIndex()):
        return self.NCOLS
    
    def data(self, index, role=qt.Qt.DisplayRole):
        if not index.isValid() or \
           not (0 <= index.row() < len(self.modfuncs)):
            return qt.QVariant()
        modfunc = self.modfuncs[index.row()]
        column = index.column()
        if role == qt.Qt.DisplayRole:
            if column == self.FNAME:
                return qt.QVariant(modfunc.name)
            elif column == self.INDEPENDENTS:
                return qt.QVariant(modfunc.independents)
            elif column == self.PARAMS:
                return qt.QVariant(modfunc.parameters)
            elif column == self.DESCRIPTION:
                return qt.QVariant(modfunc.description)
            elif column == self.DEFINITION:
                return qt.QVariant(modfunc.definition)
        elif role == qt.Qt.ToolTipRole:
            if column == self.FNAME:
                return qt.QVariant(modfunc.name)
            elif column == self.INDEPENDENTS:
                return qt.QVariant(modfunc.independents)
            elif column == self.PARAMS:
                return qt.QVariant(modfunc.parameters)
            elif column == self.DESCRIPTION:
                return qt.QVariant(modfunc.long_description)
            elif column == self.DEFINITION:
                return qt.QVariant(modfunc.definition)
        return qt.QVariant()