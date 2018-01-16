'''
Created on 15 Jan 2018

@author: schilsm
'''
from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets

import pandas as pd, numpy as np
from PyQt5.Qt import QAbstractTableModel

class CreateDataSetDialog(widgets.QDialog):

    def __init__(self, parent, selected_fn):
        '''
        Constructor
        '''
        self.function = selected_fn
        
        super(CreateDataSetDialog, self).__init__(parent)
        self.setWindowTitle("Create a data set")
        main_layout = widgets.QVBoxLayout()
        self.button_box = widgets.QDialogButtonBox()
        self.button_box.addButton(widgets.QDialogButtonBox.Cancel)
        self.button_box.addButton(widgets.QDialogButtonBox.Ok)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        txt_fn = widgets.QLabel("Modelling function: " + self.function.name)
        txt_descr = widgets.QTextEdit(self.function.long_description)
        txt_descr.setReadOnly(True)
        txt_descr.setSizeAdjustPolicy(widgets.QAbstractScrollArea.AdjustToContents)
        
        gbx_x = widgets.QGroupBox()
        gbx_x.setTitle("Series")
        
        gbx_params = widgets.QGroupBox()
        gbx_params.setTitle("Parameters")
        lo_params = widgets.QVBoxLayout()
        self.tbl_params = widgets.QTableView()
        self.param_vals = ParameterValuesTableModel(self.function)
        self.tbl_params.setModel(self.param_vals)
        lo_params.addWidget(self.tbl_params)
        gbx_params.setLayout(lo_params)

        main_layout.addWidget(txt_fn)
        main_layout.addWidget(txt_descr)
        main_layout.addWidget(gbx_x)
        main_layout.addWidget(gbx_params)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)
        
    def accept(self):
        widgets.QDialog.accept(self)
        
    def reject(self):
        widgets.QDialog.reject(self)
        
        
class ParameterValuesTableModel(qt.QAbstractTableModel):
    NCOLS = 1
    PVAL, = range(NCOLS)
    
    def __init__(self, func):  
        super(ParameterValuesTableModel, self).__init__()
        self.func = func
        d = {}
        for p in self.func.parameters:
            d[p] = 0.0
        self.values = pd.DataFrame(d)
        print(self.values)
        
    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        # Implementation of super.headerData
        if role != qt.Qt.DisplayRole:
            return qt.QVariant()
        if orientation == qt.Qt.Horizontal:
            if section == self.PVAL:
                return qt.QVariant("Value")
        if orientation == qt.Qt.Vertical:
            for i in range(len(self.func.parameters)):
                if section == i:
                    return qt.QVariant(self.func.parameters[i])
        return qt.QVariant(int(section + 1))

    def rowCount(self, index=qt.QModelIndex()):
        return len(self.func.parameters) 

    def columnCount(self, index=qt.QModelIndex()):
        return self.NCOLS
    
    def data(self, index, role=qt.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self.func.parameters)):
            return qt.QVariant()
        column, row = index.column(), index.row()
        if role == qt.Qt.DisplayRole:
            return qt.QVariant(self.param_vals[row, column] )
        return qt.QVariant() 

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        if role != qt.Qt.EditRole:
            return False
        row = index.row()
        if row < 0 or row >= len(self._data.values):
            return False
        column = index.column()
        if column < 0 or column >= self._data.columns.size:
            return False
        self._data.values[row][column] = value
        self.dataChanged.emit(index, index)
        return True   
 
    def flags(self, index):
        flags = super(self.__class__,self).flags(index)
        flags |= qt.Qt.ItemIsEditable
        flags |= qt.Qt.ItemIsSelectable
        flags |= qt.Qt.ItemIsEnabled
#         flags |= qt.Qt.ItemIsDragEnabled
#         flags |= qt.Qt.ItemIsDropEnabled
        return flags        
                        