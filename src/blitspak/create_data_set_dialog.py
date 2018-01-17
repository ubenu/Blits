'''
Created on 15 Jan 2018

@author: schilsm
'''
from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets

import pandas as pd, numpy as np
#from PyQt5.Qt import QAbstractTableModel

class CreateDataSetDialog(widgets.QDialog):

    def __init__(self, parent, selected_fn):
        self.function = selected_fn
        self.all_series = []
        self.series_names = []
        
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
        self.lo_series = widgets.QVBoxLayout()
                        
        btn_add = widgets.QPushButton("Add series")
        btn_add.clicked.connect(self.add_series)
        self.lo_series.addWidget(btn_add)
        gbx_x.setLayout(self.lo_series)
        
        self.add_series()
        
        gbx_params = widgets.QGroupBox()
        gbx_params.setTitle("Parameters")
        self.tbl_params = widgets.QTableView()
        self.param_vals = ParameterValuesTableModel(self.function)
        self.tbl_params.setModel(self.param_vals)
        self.tbl_params.setSizeAdjustPolicy(widgets.QAbstractScrollArea.AdjustToContents)
        lo_params = widgets.QVBoxLayout()
        lo_params.addWidget(self.tbl_params)
        gbx_params.setLayout(lo_params)

        main_layout.addWidget(txt_fn)
        main_layout.addWidget(txt_descr)
        main_layout.addWidget(gbx_x)
        main_layout.addWidget(gbx_params)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)
        
    def add_series(self):
        n = len(self.all_series)
        lbl = widgets.QLabel("Series name")
        txt = widgets.QLineEdit("Series " + str(n + 1))
        self.series_names.append(txt)
        hlo = widgets.QHBoxLayout()
        hlo.addWidget(lbl)
        hlo.addWidget(txt)
        tbl_series = widgets.QTableView()
        tbl_series.setSizeAdjustPolicy(widgets.QAbstractScrollArea.AdjustToContents)
        series_x = SeriesTableModel(self.function)
        self.all_series.append(series_x)
        tbl_series.setModel(series_x)
        vlo = widgets.QVBoxLayout()
        vlo.addLayout(hlo)
        vlo.addWidget(tbl_series)
        self.lo_series.addLayout(vlo)
        
    def accept(self):
        for i in range(len(self.all_series)):
            print(self.series_names[i].text())
            print(self.all_series[i].series_info)
            
        print(self.param_vals.pvalues)
        widgets.QDialog.accept(self)
        
    def reject(self):
        widgets.QDialog.reject(self)
        
class SeriesTableModel(qt.QAbstractTableModel):
    
    def __init__(self, func):  
        super(SeriesTableModel, self).__init__()
        self.func = func
        cols = ['Min', 'Max', 'Step']
        d = np.zeros((len(self.func.independents), len(cols)))
        self.series_info = pd.DataFrame(d, columns=cols, index=self.func.independents)
        
    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        # Implementation of super.headerData
        if role == qt.Qt.DisplayRole:
            if orientation == qt.Qt.Horizontal:
                return self.series_info.columns[section]
            elif orientation == qt.Qt.Vertical:
                return self.series_info.index[section]
            return qt.QVariant()
        return qt.QVariant()

    def rowCount(self, index=qt.QModelIndex()):
        return self.series_info.shape[0]

    def columnCount(self, index=qt.QModelIndex()):
        return self.series_info.shape[1]
    
    def data(self, index, role=qt.Qt.DisplayRole):
        if index.isValid():
            if role == qt.Qt.DisplayRole:
                return str(self.series_info.iloc[index.row(), index.column()])
            return qt.QVariant()
        return qt.QVariant()

    def setData(self, index, value, role):
        if index.isValid() and role == qt.Qt.EditRole:
            row, col = index.row(), index.column()
            if row in range(self.series_info.shape[0]) and col in range(self.series_info.shape[1]):
                try:
                    self.series_info.iloc[row][col] = value
                    self.dataChanged.emit(index, index)
                    return True
                except ValueError:
                    return False
            return False
        return False
 
    def flags(self, index):
        flags = super(self.__class__,self).flags(index)
        flags |= qt.Qt.ItemIsEditable
        flags |= qt.Qt.ItemIsSelectable
        flags |= qt.Qt.ItemIsEnabled
        return flags         
        
class ParameterValuesTableModel(qt.QAbstractTableModel):
    
    def __init__(self, func):  
        super(ParameterValuesTableModel, self).__init__()
        self.func = func
        d = {'Value': np.zeros(len(self.func.parameters))}
        self.pvalues = pd.DataFrame(d, self.func.parameters)
        
    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        # Implementation of super.headerData
        if role == qt.Qt.DisplayRole:
            if orientation == qt.Qt.Horizontal:
                return self.pvalues.columns[section]
            elif orientation == qt.Qt.Vertical:
                return self.pvalues.index[section]
            return qt.QVariant()
        return qt.QVariant()

    def rowCount(self, index=qt.QModelIndex()):
        return self.pvalues.shape[0]

    def columnCount(self, index=qt.QModelIndex()):
        return self.pvalues.shape[1]
    
    def data(self, index, role=qt.Qt.DisplayRole):
        if index.isValid():
            if role == qt.Qt.DisplayRole:
                return str(self.pvalues.iloc[index.row(), index.column()])
            return qt.QVariant()
        return qt.QVariant()

    def setData(self, index, value, role):
        if index.isValid() and role == qt.Qt.EditRole:
            row, col = index.row(), index.column()
            if row in range(self.pvalues.shape[0]) and col in range(self.pvalues.shape[1]):
                try:
                    self.pvalues.iloc[row][col] = value
                    self.dataChanged.emit(index, index)
                    return True
                except ValueError:
                    return False
            return False
        return False
 
    def flags(self, index):
        flags = super(self.__class__,self).flags(index)
        flags |= qt.Qt.ItemIsEditable
        flags |= qt.Qt.ItemIsSelectable
        flags |= qt.Qt.ItemIsEnabled
        return flags      
    
                        