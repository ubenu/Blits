'''
Created on 9 Jan 2018

@author: schilsm
'''

from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets

import pandas as pd, numpy as np


NCOLS = 5
FNAME, INDEPENDENTS, PARAMS, DESCRIPTION, DEFINITION = range(NCOLS)


if __name__ == '__main__':
    pass


class FunctionSelectionDialog(widgets.QDialog):
    
    def __init__(self, selected_fn_name="", parent=None):
        super(FunctionSelectionDialog, self).__init__(parent)
        
        self.model = FunctionLibrary()
        
        self.setWindowTitle("Select modelling function")
        main_layout = widgets.QVBoxLayout()
        self.button_box = widgets.QDialogButtonBox()
        self.button_box.addButton(widgets.QDialogButtonBox.Ok)
        self.button_box.addButton(widgets.QDialogButtonBox.Cancel)
        self.button_box.clicked.connect(self.on_ok)
        self.button_box.clicked.connect(self.on_cancel)
         
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
        
    def on_ok(self, button):
        if button == self.button_box.button(widgets.QDialogButtonBox.Ok):
            row = self.tableview.selectionModel().currentIndex().row()
            col = FNAME
            item = self.tableview.model().index(row, col, qt.QModelIndex())
            self.selected_fn_name = self.tableview.model().data(item).value()
            self.accept()

    def on_cancel(self, button):
        if button == self.button_box.button(widgets.QDialogButtonBox.Cancel):
            self.reject()
          
 
class ModellingFunction(object):
    
    def __init__(self, uid):
        """
        @uid: unique identifier
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

 
class FunctionLibrary(qt.QAbstractTableModel):
    
    def __init__(self, filepath="..\\..\\Resources\\ModellingFunctions\\Functions.csv"):
        super(FunctionLibrary, self).__init__()
        
        self.filepath = filepath
        self.raw_data = None
        self.dirty = False
        
        self.load_lib()
        
    
    def load_lib(self):    
        self.modfuncs = []
        
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
            modfunc.parameters = par
            modfunc.first_estimates = est
            self.modfuncs.append(modfunc)
            
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
        # Implementation of super.rowCount
        return len(self.modfuncs)


    def columnCount(self, index=qt.QModelIndex()):
        # Implementation of super.columnCount
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
