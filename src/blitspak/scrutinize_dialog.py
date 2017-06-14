'''
Created on 6 Jun 2017

@author: SchilsM
'''

import sys
import numpy as np
import copy as cp
from scipy.optimize import curve_fit

from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets
from blitspak.blits_mpl import MplCanvas, NavigationToolbar
#from blitspak.blits_data import BlitsData as bld

import functions.framework as ff
import functions.function_defs as fdefs

#import scrutinize_dialog_ui as ui

from PyQt5.uic import loadUiType
#from functions.framework import FunctionsFramework
Ui_ScrutinizeDialog, QDialog = loadUiType('..\\..\\Resources\\UI\\scrutinize_dialog.ui')

class ScrutinizeDialog(widgets.QDialog, Ui_ScrutinizeDialog):
    
    available_functions = range(5)
    f_avg, f_lin, f_ex1, f_ex2, f_ex3 = available_functions
    fn_names = {f_avg: "Average",
                f_lin: "Straight line",
                f_ex1: "Single exponential",
                f_ex2: "Double exponential",
                f_ex3: "Triple exponential",
                }

    fd_fields = range(4)
    d_func, d_p0, d_pnames, d_expr = fd_fields
    fn_dictionary = {"Average": (fdefs.fn_average, 
                                 fdefs.p0_fn_average,
                                 ('a',), 
                                 "average(trace)"),
                     "Straight line": (fdefs.fn_straight_line, 
                                       fdefs.p0_fn_straight_line,
                                       ('a', 'b'), 
                                       "a + b.x"), 
                     "Single exponential": (fdefs.fn_1exp, 
                                            fdefs.p0_fn_1exp,
                                            ('a0', 'a1', 'k1'), 
                                            "a0 + a1.exp(-x.k1)"),
                     "Double exponential": (fdefs.fn_2exp, 
                                            fdefs.p0_fn_2exp,
                                            ('a0', 'a1', 'k1', 'a2', 'k2'), 
                                            "a0 + a1.exp(-x.k1) + a2..exp(-x.k2)"), 
                     "Triple exponential": (fdefs.fn_3exp, 
                                            fdefs.p0_fn_3exp,
                                            ('a0', 'a1', 'k1', 'a2', 'k2', 'a3', 'k3'), 
                                            "a0 + a1.exp(-x.k1) + a2.exp(-x.k2) + a3.exp(-x.k3)"),
                     "Michaelis-Menten equation": (fdefs.fn_mich_ment,
                                                   fdefs.p0_fn_mich_ment,
                                                   ('km', 'vmax'), 
                                                   "vmax . x / (km + x)"),
                     "Competitive inhibition equation": (fdefs.fn_comp_inhibition, 
                                                         fdefs.p0_fn_comp_inhibition,
                                                         ('km', 'ki', 'vmax'), 
                                                         "vmax . x[0] / (km . (1.0 + x[1] / ki) + x[0])"), 
                     "Hill equation": (fdefs.fn_hill, 
                                       fdefs.p0_fn_hill,
                                       ('ymax', 'xhalf', 'h'), 
                                       "ymax / ((xhalf/x)^h + 1 )"),
                     }
    
    params_table_columns = range(4)
    hp_trac, hp_p0, hp_cons, hp_link = params_table_columns
    params_table_headers = {hp_trac: "Trace",
                            hp_p0: "Init est",
                            hp_cons: "Const",
                            hp_link: "Linked",
                            }
    results_table_columns = range(4)
    hr_trac, hr_pfit, hr_conf, hr_advice = results_table_columns
    results_table_headers = {hr_trac: "Trace",
                             hr_pfit: "Value",
                             hr_conf: "Confidence\ninterval",
                             hr_advice: "",
                             }

    def __init__(self, parent, start, stop):
        '''
        Constructor
        '''
        self.ui_ready = False
        super(widgets.QDialog, self).__init__(parent)
        self.setupUi(self)
        ## Create the plot area
        self.canvas = MplCanvas(self.mpl_window)
        self.mpl_layout.addWidget(self.canvas)
        self.plot_toolbar = NavigationToolbar(self.canvas, self.mpl_window)
        self.mpl_layout.addWidget(self.plot_toolbar)
        ## Connect signals to slots
        self.cmb_fit_function.currentIndexChanged.connect(self.on_current_index_changed)
        self.tbl_params.itemChanged.connect(self.on_item_changed)
        self.btn_calc.clicked.connect(self.on_calc)
        self.on_current_index_changed(self.cmb_fit_function.currentIndex())
        ## Transfer the fitting functions from the (temporary) dictionary to ModellingFunction objects
        self.library = {}
        self.fill_library()
        self.current_function = ""
        # Prepare the UI
        self.cmb_fit_function.setSizeAdjustPolicy(widgets.QComboBox.AdjustToContents)
        for i in self.available_functions:
            name = self.fn_names[i]
            self.cmb_fit_function.addItem(name)

        self.fnfrw = ff.FunctionsFramework()  
        self.display_curves = None
        ## Add the data and draw them
        indmin, indmax = np.searchsorted(self.parent().blits_data.working_data['time'],(start, stop))
        self.data = cp.deepcopy(self.parent().blits_data.working_data[indmin:indmax])
        self.trace_ids = self.data.columns[self.data.columns != 'time']
        self.draw_data()
        
        self.ui_ready = True
        self.on_current_index_changed(0)
                
    def on_calc(self):
        nfunc = self.cmb_fit_function.currentText()
        f = self.fn_dictionary[nfunc][self.d_func]
        selection = self.data 
        self.display_curves = cp.deepcopy(selection)
        self.display_curves[self.trace_ids] = np.zeros_like(selection[self.trace_ids])
        p0s = self.get_p0s_from_table()
        for trace in self.trace_ids:
            p = p0s[trace]
            t = selection['time']
            x = t - t.iloc[0]
            y = selection[trace]
            try:
                pfit, pcov = curve_fit(self.fnfrw.make_func(f, params=p, const={}), x, y, p0=p)
                pconf = self.fnfrw.confidence_intervals(y.shape[0], pfit, pcov, 0.95)
                prelc = pfit / pconf
                self.pfits[trace] = pfit
                self.pconfs[trace] = pfit
                self.prelcs[trace] = prelc
                self.display_curves[trace] = self.fnfrw.display_curve(f, x, pfit)
            except ValueError as e:
                print(e)
            except RuntimeError as e:  
                print(e)
            except TypeError as e:
                print(e)               
            except:
                e = sys.exc_info()[0]
                print(e)
        self.draw_data()
    self.prepare_results_table()   
         
    def on_current_index_changed(self, index):
        self.pfits, self.pconfs, self.prelcs = {}, {}, {}
        if self.ui_ready:
            self.txt_function.clear()
            self.current_function = self.cmb_fit_function.currentText()
            self.txt_function.setText(self.library[self.current_function].fn_str)
            self.txt_function.adjustSize()
            self.prepare_params_table()
            self.prepare_results_table()
        
    def on_item_changed(self, item):
        col, row = item.column(), item.row()
        if row == 0 and (col - 1) % 3 in (1, 2):
            cs = item.checkState()
            for irow in range(1,self.tbl_params.rowCount()):
                w = self.tbl_params.item(irow, col)
                if not w is None:
                    w.setCheckState(cs)
                    
    def prepare_results_table(self):
        self.tbl_results.clear()
        labels = [self.results_table_headers[self.hr_trac],]
        if self.current_function != "":
            pnames = self.library[self.current_function].param_names 
            for name in pnames:
                labels.append(self.results_table_headers[self.hr_pfit] + '\n' + name)
                labels.append(self.results_table_headers[self.hr_conf])
                labels.append(self.results_table_headers[self.hr_advice])
        self.tbl_results.setColumnCount(len(labels))
        self.tbl_results.setHorizontalHeaderLabels(labels)
        if len(self.trace_ids) != 0:
            labels = []
            labels.extend(self.trace_ids.tolist())
            self.tbl_results.setRowCount(len(labels))
            self.tbl_results.setVerticalHeaderLabels(labels)
            self.tbl_results.resizeColumnsToContents()
            self.tbl_results.resizeRowsToContents()            
        for irow in range(self.tbl_results.rowCount()):
            cid = self.tbl_results.verticalHeaderItem(irow).text()
            for icol in range(self.tbl_results.columnCount()):
                if icol == 0: # curve colour icon in col 0
                    w = widgets.QTableWidgetItem()
                    col = self.canvas.curve_colours[cid]
                    ic = self.parent().line_icon(col)
                    w.setIcon(ic)
                    self.tbl_results.setItem(irow, icol, w)   
                elif icol % 4 in (0, 1, 2):
                    pass
                                
    def prepare_params_table(self):
        self.tbl_params.clear()
        labels = [self.params_table_headers[self.hp_trac],]
        if self.current_function != "":
            pnames = self.library[self.current_function].param_names 
            for name in pnames:
                labels.append(self.params_table_headers[self.hp_p0] + '\n' + name)
                labels.append(self.params_table_headers[self.hp_cons])
                labels.append(self.params_table_headers[self.hp_link])
        self.tbl_params.setColumnCount(len(labels))
        self.tbl_params.setHorizontalHeaderLabels(labels)
        if len(self.trace_ids) != 0:
            labels = ['All',]
            labels.extend(self.trace_ids.tolist())
            self.tbl_params.setRowCount(len(labels))
            self.tbl_params.setVerticalHeaderLabels(labels)
            self.tbl_params.resizeColumnsToContents()
            self.tbl_params.resizeRowsToContents()
        
        p0s = self.get_p0s() 
        for irow in range(self.tbl_params.rowCount()):
            tid = self.trace_ids[irow-1]
            p0 = p0s[tid]
            for icol in range(self.tbl_params.columnCount()):
                if icol == 0 and irow != 0: # curve colour icon in col 0
                    w = widgets.QTableWidgetItem()
                    cid = self.tbl_params.verticalHeaderItem(irow).text()
                    col = self.canvas.curve_colours[cid]
                    ic = self.parent().line_icon(col)
                    w.setIcon(ic)
                    self.tbl_params.setItem(irow, icol, w)
                elif (icol - 1) % 3 in (1, 2): # checkboxes in col 1 and 2 for each param
                    w = widgets.QTableWidgetItem() 
                    w.setCheckState(qt.Qt.Unchecked)
                    if irow != 0 or icol != 0: # (icol - 1) % 3 in (1, 2) and icol != 0:
                        self.tbl_params.setItem(irow, icol, w)
                elif (icol - 1) % 3 in (0, ) and irow != 0:
                    w = widgets.QTableWidgetItem()
                    np = (icol - 1) // 3
                    w.setText('{:.2g}'.format(p0[np]))
                    self.tbl_params.setItem(irow, icol, w)
                    
    def get_p0s(self):
        p0_func = self.library[self.current_function].p0_fn_ref
        x = self.data['time']
        p0s = {}
        for tid in self.trace_ids:
            y = self.data[tid]
            p0s[tid] = p0_func(x, y)
        return p0s
                               
    def get_p0s_from_table(self):
        p0s = {}
        for irow in range(self.tbl_params.rowCount()):
            tid = self.trace_ids[irow-1]
            p0 = []
            for icol in range(self.tbl_params.columnCount()):
                if (icol - 1) % 3 in (0, ) and irow != 0:
                    txt = self.tbl_params.item(irow, icol).text()
                    p0.append(float(txt))
            p0s[tid] = np.array(p0) 
        return p0s
        
                     
    def on_check_state_changed(self):
        pass
                        
    def draw_data(self):
        if not self.data is None:
            x = self.data['time']
            y = self.data[self.trace_ids]
            self.canvas.draw_data(x, y)
        if not self.display_curves is None:
            xd = self.display_curves['time']
            yd = self.display_curves[self.trace_ids]
            self.canvas.draw_fitted_data(xd, yd)
            

    def fill_library(self):                
        for name in self.fn_dictionary:
            fn_ref = self.fn_dictionary[name][self.d_func]
            p0_fn_ref = self.fn_dictionary[name][self.d_p0]
            param_names = self.fn_dictionary[name][self.d_pnames]
            fn_str = self.fn_dictionary[name][self.d_expr]
            self.library[name] = ff.ModellingFunction(name, fn_ref, p0_fn_ref, param_names, fn_str)    
            
        