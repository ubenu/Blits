'''
Created on 15 Jan 2018

@author: schilsm
'''
from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets

import pandas as pd, numpy as np
from scipy.stats import norm
import copy as cp

from blitspak.crux_table_model import CruxTableModel
#from numba.tests.npyufunc.test_ufunc import dtype
#from functions.function_defs import n_independents

class DataCreationDialog(widgets.QDialog):

    def __init__(self, parent, selected_fn):
        self.function = selected_fn
        self.all_series_names = []
        self.series_params = {}
        self.series_axes_info = {}
        self.template = None
        
        super(DataCreationDialog, self).__init__(parent)
        self.setWindowTitle("Create a data set")
        
        # Buttonbox
        main_layout = widgets.QVBoxLayout()
        self.button_box = widgets.QDialogButtonBox()
        self.btn_add_series = widgets.QPushButton("Add series")
        self.button_box.addButton(self.btn_add_series, widgets.QDialogButtonBox.ActionRole)
        self.button_box.addButton(widgets.QDialogButtonBox.Cancel)
        self.button_box.addButton(widgets.QDialogButtonBox.Ok)
        self.btn_add_series.clicked.connect(self.add_series)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        # Function description widgets
        txt_fn = widgets.QLabel("Modelling function: " + self.function.name)
        txt_descr = widgets.QTextEdit(self.function.long_description)
        txt_descr.setReadOnly(True)
        txt_descr.setSizeAdjustPolicy(widgets.QAbstractScrollArea.AdjustToContents)
        
        # Series tab widget
        self.tab_series = widgets.QTabWidget()
        self.add_series()
        
        # Main layout
        main_layout.addWidget(txt_fn)
        main_layout.addWidget(txt_descr)
        main_layout.addWidget(self.tab_series)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)
        
    def add_series(self):
        try:
            n = len(self.all_series_names)
            name = "Series " + str(n + 1)
            self.all_series_names.append(name)
    
            lbl_n = widgets.QLabel("Number of data points")
            txt_n = widgets.QLineEdit("21")
            lbl_std = widgets.QLabel("Noise on data (StDev)")
            txt_std = widgets.QLineEdit("0.0")
            
            lbl_inds = widgets.QLabel('Axes')
            indx_inds = self.function.independents
            cols_inds = ['Start', 'End']
            extremes = np.zeros((len(indx_inds), len(cols_inds)), dtype=float)
            df_inds = pd.DataFrame(extremes, index=indx_inds, columns=cols_inds)
            df_inds.End = 1.0
            lbl_pars = widgets.QLabel('Parameters')
            indx_pars = self.function.parameters
            cols_pars = [name]
            df_pars = pd.DataFrame(np.ones((len(indx_pars), len(cols_pars)), dtype=float), index=indx_pars, columns=cols_pars)
            
            mdl_inds = CruxTableModel(df_inds)
            self.series_axes_info[name] = (mdl_inds, txt_n, txt_std)
            tbl_inds = widgets.QTableView()
            tbl_inds.setModel(mdl_inds)
            tbl_inds.setSizeAdjustPolicy(widgets.QAbstractScrollArea.AdjustToContents)
            mdl_pars = CruxTableModel(df_pars)
            self.series_params[name] = mdl_pars
            tbl_pars = widgets.QTableView()
            tbl_pars.setModel(mdl_pars)
            tbl_pars.setSizeAdjustPolicy(widgets.QAbstractScrollArea.AdjustToContents)
    
            w = widgets.QWidget()
            glo1 = widgets.QGridLayout()
            glo1.addWidget(lbl_n, 0, 0, alignment=qt.Qt.AlignHCenter)
            glo1.addWidget(lbl_std, 0, 1, alignment=qt.Qt.AlignHCenter)
            glo1.addWidget(txt_n, 1, 0, alignment=qt.Qt.AlignHCenter)
            glo1.addWidget(txt_std, 1, 1, alignment=qt.Qt.AlignHCenter)
            glo1.addWidget(lbl_inds, 2, 0, alignment=qt.Qt.AlignHCenter)
            glo1.addWidget(lbl_pars, 2, 1, alignment=qt.Qt.AlignHCenter)
            glo1.addWidget(tbl_inds, 3, 0, alignment=qt.Qt.AlignHCenter)
            glo1.addWidget(tbl_pars, 3, 1, alignment=qt.Qt.AlignHCenter)
            vlo = widgets.QVBoxLayout()
            vlo.addLayout(glo1)
            
            w.setLayout(vlo)
            self.tab_series.addTab(w, name)
            self.tab_series.setCurrentWidget(w)
            
        except Exception as e:
            print(e)
            
    def create_template(self):
        df_all_series = pd.DataFrame()
        df_all_parameters = pd.DataFrame()
        for name in self.all_series_names:
            df_p = self.series_params[name].df_data # parameters for this series
            df_all_parameters = pd.concat([df_all_parameters, df_p], axis=1) # add to all parameters
            
            df_si = self.series_axes_info[name][0].df_data # axes start, stop, and std on data
            n = int(self.series_axes_info[name][1].text()) # number of points in series
            std = float(self.series_axes_info[name][2].text())
            cols = df_si.index # independent axes names
            df_s = pd.DataFrame([], index=range(n), columns=cols) # dataframe for axes values
            for col in cols:
                df_s[col] = np.linspace(df_si.iloc[:,0][col], df_si.iloc[:,1][col], n)
            x = cp.deepcopy(df_s).as_matrix().transpose() # copy axes values and transpose for use in self.function
            params = cp.deepcopy(df_p).as_matrix()
            vals = pd.DataFrame([], index=range(n), columns=[name]) # dataframe for dependent values
            y = self.function.func(x, params)
            if std > 0:
                y = norm.rvs(loc=y, scale=std)
            vals[name] = y
            
            df_s = pd.concat([df_s, vals], axis=1)
            df_all_series = pd.concat([df_all_series, df_s], axis=1)
            
        return (df_all_series, df_all_parameters, self.function)
        
    def accept(self):
        self.template = self.create_template()
        widgets.QDialog.accept(self)
        
    def reject(self):
        widgets.QDialog.reject(self)
        
        
# class DataSet(pd.DataFrame):
#     
#     def __init__(self):
#         self.series = {}
#         
#     def add_series(self, series):
#         if series.name() in self.series:
#             print("Existing series overwritten")
#         self.series[series.name()] = series
#         
#         
# class DataSeries():
#     
#     def __init__(self, name, independents, dependent, ind_names=['x', ]):
#         header = ind_names.extend([name])
#         self.data = pd.DataFrame(np.vstack(independents, dependent), columns=header)
#         
#     def axes(self):
#         return cp.deepcopy(self.data.iloc[:, :-1])
# 
#     def axes_names(self):
#         return cp.deepcopy(self.data.columns.tolist()[:-1])
#     
#     def values(self):
#         return cp.deepcopy(self.data.iloc[:, -1])
#     
#     def name(self):
#         return cp.deepcopy(self.data.columns.tolist[-1])
#     
#     def set_name(self, name):
#         cols = self.data.columns.tolist()
#         cols[-1] = name
#         self.data.columns = cols
#         
        
        
        