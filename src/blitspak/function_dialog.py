'''
Created on 9 Jan 2018

@author: schilsm
'''

from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets

import pandas as pd, numpy as np

import functions.function_defs as fdefs

NCOLS = 5
FNAME, INDEPENDENTS, PARAMS, DESCRIPTION, DEFINITION = range(NCOLS)


if __name__ == '__main__':
    pass



class FunctionSelectionDialog(widgets.QDialog):
    
    def __init__(self, parent, selected_fn_name=""):
        super(FunctionSelectionDialog, self).__init__(parent)
        self.setModal(False)
        
        self.model = FunctionLibraryTableModel()
        
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
        self.tableview.setModel(self.model)
        self.tableview.setSelectionBehavior(widgets.QAbstractItemView.SelectRows)
        self.tableview.setSelectionMode(widgets.QAbstractItemView.SingleSelection)
        
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
        row, col = self.tableview.selectionModel().currentIndex().row(), FNAME
        inx = self.tableview.model().index(row, col, qt.QModelIndex())
        self.selected_fn_name = self.tableview.model().data(inx).value()
     
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
        self.parameters = ""
        self.first_estimates = ""
        self.func = None 
        self.p0 = None       

 
class FunctionLibraryTableModel(qt.QAbstractTableModel):

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
        
    def __init__(self, filepath="..\\..\\Resources\\ModellingFunctions\\Functions.csv"):
        super(FunctionLibraryTableModel, self).__init__()
        
        self.filepath = filepath
        self.raw_data = None
        self.dirty = False
        
        self.modfuncs = []
        self.funcion_dictionary = {}
        self.load_lib()        
    
    def load_lib(self):    
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
            modfunc.obs_dependent_name = odp
            modfunc.calc_dependent_name = cdp
            modfunc.independents = idp 
            modfunc.parameters = par.strip().split(',')
            modfunc.first_estimates = est
            modfunc.func = self.fn_dictionary[modfunc.name][self.M_FUNC]
            modfunc.p0 = self.fn_dictionary[modfunc.name][self.M_P0]
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
            if section == FNAME:
                return qt.QVariant("Name")
            elif section == INDEPENDENTS:
                return qt.QVariant("Independents")
            elif section == PARAMS:
                return qt.QVariant("Parameters")
            elif section == DESCRIPTION:
                return qt.QVariant("Description")
            elif section == DEFINITION:
                return qt.QVariant("Definition")
        return qt.QVariant(int(section + 1))

    def rowCount(self, index=qt.QModelIndex()):
        return len(self.modfuncs)

    def columnCount(self, index=qt.QModelIndex()):
        return NCOLS
    
    def data(self, index, role=qt.Qt.DisplayRole):
        if not index.isValid() or \
           not (0 <= index.row() < len(self.modfuncs)):
            return qt.QVariant()
        modfunc = self.modfuncs[index.row()]
        column = index.column()
        if role == qt.Qt.DisplayRole:
            if column == FNAME:
                return qt.QVariant(modfunc.name)
            elif column == INDEPENDENTS:
                return qt.QVariant(modfunc.independents)
            elif column == PARAMS:
                return qt.QVariant(modfunc.parameters)
            elif column == DESCRIPTION:
                return qt.QVariant(modfunc.description)
            elif column == DEFINITION:
                return qt.QVariant(modfunc.definition)
        elif role == qt.Qt.ToolTipRole:
            if column == FNAME:
                return qt.QVariant(modfunc.name)
            elif column == INDEPENDENTS:
                return qt.QVariant(modfunc.independents)
            elif column == PARAMS:
                return qt.QVariant(modfunc.parameters)
            elif column == DESCRIPTION:
                return qt.QVariant(modfunc.long_description)
            elif column == DEFINITION:
                return qt.QVariant(modfunc.definition)
        return qt.QVariant()

       