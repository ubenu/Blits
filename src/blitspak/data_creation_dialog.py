'''
Created on 15 Jan 2018

@author: schilsm
'''
from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets

import pandas as pd, numpy as np
import copy as cp

from blitspak.crux_table_model import CruxTableModel
from numba.tests.npyufunc.test_ufunc import dtype
from functions.function_defs import n_independents

class DataCreationDialog(widgets.QDialog):

    def __init__(self, parent, selected_fn):
        self.function = selected_fn
        self.all_series = []
        self.all_series_names = []
        self.series_names = []
        
        super(DataCreationDialog, self).__init__(parent)
        self.setWindowTitle("Create a data set")
        main_layout = widgets.QVBoxLayout()
        self.button_box = widgets.QDialogButtonBox()
        self.btn_add_series = widgets.QPushButton("Add series")
        self.btn_create_template = widgets.QPushButton("Create template")
        self.button_box.addButton(self.btn_add_series, widgets.QDialogButtonBox.ActionRole)
        self.button_box.addButton(self.btn_create_template, widgets.QDialogButtonBox.ActionRole)
        self.button_box.addButton(widgets.QDialogButtonBox.Cancel)
        self.button_box.addButton(widgets.QDialogButtonBox.Ok)
        self.btn_add_series.clicked.connect(self.add_series)
        self.btn_create_template.clicked.connect(self.create_template)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        txt_fn = widgets.QLabel("Modelling function: " + self.function.name)
        txt_descr = widgets.QTextEdit(self.function.long_description)
        txt_descr.setReadOnly(True)
        txt_descr.setSizeAdjustPolicy(widgets.QAbstractScrollArea.AdjustToContents)
        
        gbx_series = widgets.QGroupBox()
        gbx_series.setTitle("Series - axes")
        self.lo_series = widgets.QVBoxLayout()
        gbx_series.setLayout(self.lo_series)
        
        self.add_series()
        
        gbx_params = widgets.QGroupBox()
        gbx_params.setTitle("Parameters")
        self.tbl_params = widgets.QTableView()
        cols = ['Value', ]
        d = np.zeros((len(self.function.parameters), len(cols)))
        df = pd.DataFrame(d, columns=cols, index=self.function.parameters)
        self.param_vals = CruxTableModel(df)

        self.tbl_params.setModel(self.param_vals)
        self.tbl_params.setSizeAdjustPolicy(widgets.QAbstractScrollArea.AdjustToContents)
        lo_params = widgets.QVBoxLayout()
        lo_params.addWidget(self.tbl_params)
        gbx_params.setLayout(lo_params)

        main_layout.addWidget(txt_fn)
        main_layout.addWidget(txt_descr)
        main_layout.addWidget(gbx_series)
        main_layout.addWidget(gbx_params)
        main_layout.addWidget(self.button_box)
        
        self.setLayout(main_layout)
        
    def add_series(self):
        n = len(self.all_series_names)
        name = "Series " + str(n + 1)
        self.all_series_names.append(name)
        lbl_name = widgets.QLabel("Series name")        
        txt_name = widgets.QLineEdit(name)
        lbl_n = widgets.QLabel("Number of points in series")
        txt_n = widgets.QLineEdit("21")
        indx = self.function.independents
        cols = ['Axis', 'Start point', 'End point']
        df = pd.DataFrame(np.zeros((len(indx), len(cols)), dtype=float), index=indx, columns=cols)
        mdl = CruxTableModel(df)
        tbl = widgets.QTableView()
        tbl.setModel(mdl)
        tbl.setSizeAdjustPolicy(widgets.QAbstractScrollArea.AdjustToContents)
        hlo1 = widgets.QHBoxLayout()
        hlo1.addWidget(lbl_name)
        hlo1.addWidget(txt_name)
        hlo1.addWidget(lbl_n)
        hlo1.addWidget(txt_n)
        vlo = widgets.QVBoxLayout()
        vlo.addLayout(hlo1)
        vlo.addWidget(tbl)
        gbx = widgets.QGroupBox(name)
        gbx.setLayout(vlo)
        self.lo_series.addWidget(gbx)
        
    def create_template(self):
        series = widgets.QInputDialog.getInt(self, "Number of series", "Number of series", 1, 1, 20, 1)
        if series[1]: # True if accepted
            n_series = series[0]
            n_independents = len(self.function.independents)
            n_params = len(self.function.parameters)
            p_cols = ["Series "  + str(i + 1) for i in range(n_series)]
            df_params = pd.DataFrame(np.ones((n_params, n_series)), columns=p_cols, index=self.function.parameters)
            t_cols = cp.deepcopy(self.function.independents)
            t_cols.append("f")                    
            t_cols *= n_series
            t_lvls = []
            for i in p_cols:
                t_lvls.extend([i] * (n_independents + 1))
            df_series = pd.DataFrame(np.full((10, len(t_cols)), np.nan, dtype=float))
            df_series = pd.concat([pd.DataFrame(t_lvls).transpose(), pd.DataFrame(t_cols).transpose(), df_series], ignore_index=True)
            
            file_path = widgets.QFileDialog.getSaveFileName(
                None,
                "Save data template to Excel",
                "DataTemplate.xlsx",
                "Excel X files (*.xlsx);;All files (*.*)")
            if file_path[0] != "":
                try:
                    writer = pd.ExcelWriter(file_path[0], engine="xlsxwriter")
                    df_series.to_excel(writer, 'Series', index=False)
                    df_params.to_excel(writer, 'Parameters')
                    writer.save()
                    writer.close()
                except Exception as e:
                    print(e)
                    msg = widgets.QMessageBox(self)
                    msg.setText("Error") 
                    msg.exec()
        
    def accept(self):
        for i in range(len(self.all_series)):
            print(self.series_names[i].text())
            print(self.all_series[i].df_data)
            
        print(self.param_vals.df_data)
        widgets.QDialog.accept(self)
        
    def reject(self):
        widgets.QDialog.reject(self)
        
# class SeriesTableModel(qt.QAbstractTableModel):
#     
#     def __init__(self, func):  
#         super(SeriesTableModel, self).__init__()
#         self.function = func
#         cols = ['Min', 'Max', 'Step']
#         d = np.zeros((len(self.function.independents), len(cols)))
#         self.series_info = pd.DataFrame(d, columns=cols, index=self.function.independents)
#         
#     def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
#         # Implementation of super.headerData
#         if role == qt.Qt.DisplayRole:
#             if orientation == qt.Qt.Horizontal:
#                 return self.series_info.columns[section]
#             elif orientation == qt.Qt.Vertical:
#                 return self.series_info.index[section]
#             return qt.QVariant()
#         return qt.QVariant()
# 
#     def rowCount(self, index=qt.QModelIndex()):
#         return self.series_info.shape[0]
# 
#     def columnCount(self, index=qt.QModelIndex()):
#         return self.series_info.shape[1]
#     
#     def data(self, index, role=qt.Qt.DisplayRole):
#         if index.isValid():
#             if role == qt.Qt.DisplayRole:
#                 return str(self.series_info.iloc[index.row(), index.column()])
#             return qt.QVariant()
#         return qt.QVariant()
# 
#     def setData(self, index, value, role):
#         if index.isValid() and role == qt.Qt.EditRole:
#             row, col = index.row(), index.column()
#             if row in range(self.series_info.shape[0]) and col in range(self.series_info.shape[1]):
#                 try:
#                     self.series_info.iloc[row][col] = value
#                     self.dataChanged.emit(index, index)
#                     return True
#                 except ValueError:
#                     return False
#             return False
#         return False
#  
#     def flags(self, index):
#         flags = super(self.__class__,self).flags(index)
#         flags |= qt.Qt.ItemIsEditable
#         flags |= qt.Qt.ItemIsSelectable
#         flags |= qt.Qt.ItemIsEnabled
#         return flags         
#         
# class ParameterValuesTableModel(qt.QAbstractTableModel):
#     
#     def __init__(self, func):  
#         super(ParameterValuesTableModel, self).__init__()
#         self.function = func
#         d = {'Value': np.zeros(len(self.function.parameters))}
#         self.pvalues = pd.DataFrame(d, self.function.parameters)
#         
#     def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
#         # Implementation of super.headerData
#         if role == qt.Qt.DisplayRole:
#             if orientation == qt.Qt.Horizontal:
#                 return self.pvalues.columns[section]
#             elif orientation == qt.Qt.Vertical:
#                 return self.pvalues.index[section]
#             return qt.QVariant()
#         return qt.QVariant()
#  
#     def rowCount(self, index=qt.QModelIndex()):
#         return self.pvalues.shape[0]
#  
#     def columnCount(self, index=qt.QModelIndex()):
#         return self.pvalues.shape[1]
#      
#     def data(self, index, role=qt.Qt.DisplayRole):
#         if index.isValid():
#             if role == qt.Qt.DisplayRole:
#                 return str(self.pvalues.iloc[index.row(), index.column()])
#             return qt.QVariant()
#         return qt.QVariant()
#  
#     def setData(self, index, value, role):
#         if index.isValid() and role == qt.Qt.EditRole:
#             row, col = index.row(), index.column()
#             if row in range(self.pvalues.shape[0]) and col in range(self.pvalues.shape[1]):
#                 try:
#                     self.pvalues.iloc[row][col] = value
#                     self.dataChanged.emit(index, index)
#                     return True
#                 except ValueError:
#                     return False
#             return False
#         return False
#   
#     def flags(self, index):
#         flags = super(self.__class__,self).flags(index)
#         flags |= qt.Qt.ItemIsEditable
#         flags |= qt.Qt.ItemIsSelectable
#         flags |= qt.Qt.ItemIsEnabled
#         return flags      

#         s = [name] * len(self.function.independents)
#         #s.extend([""] * (len(self.function.independents)-1))
        
#         cols = ['Independent', 'First\nvalue', 'Last\nvalue', 'Number\nof points']
#         d = np.zeros((len(self.function.independents), len(cols)), dtype=float)
#         ndf = pd.DataFrame(d, columns=cols, index=s)
#         ndf = ndf.astype({'Independent': object})
#         ndf.Independent =self.function.independents
#         if self.mod_x_info is None:
#             self.mod_x_info = CruxTableModel(ndf)
#         else:
#             self.mod_x_info.df_data = self.mod_x_info.df_data.append(ndf)
#         self.tbl_x_info.setModel(self.mod_x_info)
#         self.mod_x_info.layoutChanged.emit()
        
        
        
#         glo = widgets.QGridLayout()
#         row = 0
#         for i in self.function.independents:
#             lbl_ind = widgets.QLabel(i)
#             txt_start = widgets.QLineEdit("0.0")
#             txt_end = widgets.QLineEdit("100.0")
#         self.series_names.append(txt)
#         hlo = widgets.QHBoxLayout()
#         hlo.addWidget(lbl)
#         hlo.addWidget(txt)
#         tbl_series = widgets.QTableView()
#         tbl_series.setSizeAdjustPolicy(widgets.QAbstractScrollArea.AdjustToContents)
#         
#         cols = ['Series', 'First\nvalue', 'Last\nvalue', 'Step\nsize']
#         d = np.zeros((len(self.function.independents), len(cols)))
#         df = pd.DataFrame(d, columns=cols, index=self.function.independents)
#         df = df.astype({'Series': object})
#         s = [name] * (len(self.function.independents))
#         df.Series = s
#         
#         series_x = CruxTableModel(df)
#         self.all_series.append(series_x)
#         tbl_series.setModel(series_x)
#         vlo = widgets.QVBoxLayout()
#         vlo.addLayout(hlo)
#         vlo.addWidget(tbl_series)
#         self.lo_series.addLayout(vlo)    
                        