'''
Created on 6 Jun 2017

@author: SchilsM
'''

import sys
import numpy as np
import copy as cp
from scipy.optimize import curve_fit
from statsmodels.stats.stattools import durbin_watson

from PyQt5 import QtCore as qt
#from PyQt5 import QtGui as gui
from PyQt5 import QtWidgets as widgets
from blitspak.blits_mpl import MplCanvas, NavigationToolbar, DraggableLine

import functions.framework as ff
import functions.function_defs as fdefs

#import scrutinize_dialog_ui as ui

from PyQt5.uic import loadUiType
#from dill.pointers import parent # How did this get here?
#from functions.framework import FunctionsFramework
Ui_ScrutinizeDialog, QDialog = loadUiType('..\\..\\Resources\\UI\\scrutinize_dialog.ui')

class ScrutinizeDialog(widgets.QDialog, Ui_ScrutinizeDialog):
    
    available_functions = range(7)
    f_avg, f_lin, f_ex1, f_lex1, f_ex2, f_lex2, f_ex3 = available_functions
    fn_names = {f_avg: "Average",
                f_lin: "Straight line",
                f_ex1: "Single exponential",
                f_lex1: "Single exponential and straight line",
                f_ex2: "Double exponential",
                f_lex2: "Double exponential and straight line",
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
                     "Single exponential and straight line": (fdefs.fn_1exp_strline, 
                                            fdefs.p0_fn_1exp_strline,
                                            ('a0', 'a1', 'k1', 'b'), 
                                            "a0 + a1.exp(-x.k1) + b.x"), 
                     "Double exponential": (fdefs.fn_2exp, 
                                            fdefs.p0_fn_2exp,
                                            ('a0', 'a1', 'k1', 'a2', 'k2'), 
                                            "a0 + a1.exp(-x.k1) + a2..exp(-x.k2)"), 
                     "Double exponential and straight line": (fdefs.fn_2exp_strline, 
                                            fdefs.p0_fn_2exp_strline,
                                            ('a0', 'a1', 'k1', 'a2', 'k2', 'b'), 
                                            "a0 + a1.exp(-x.k1) + a2.exp(-x.k2) + b.x"), 
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
                            hp_link: "Linked\nto",
                            }
    results_table_columns = range(4)
    hr_trac, hr_dw, hr_pfit, hr_conf = results_table_columns
    results_table_headers = {hr_trac: "Trace",
                             hr_dw: "Durbin-\nWatson",
                             hr_pfit: "Value",
                             hr_conf: "Error",
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
        # Call on_current_index_changed to populate the tables
        self.on_current_index_changed(self.cmb_fit_function.currentIndex())

        ## Transfer the fitting functions from the (temporary) dictionary to ModellingFunction objects
        self.library = {}
        self.fill_library()
        self.current_function = ""
        self.param_values_fit, self.conf_intervals_fit, self.dw_statistic_fit = {}, {}, {}
        # Prepare the UI
        self.cmb_fit_function.setSizeAdjustPolicy(widgets.QComboBox.AdjustToContents)
        for i in self.available_functions:
            name = self.fn_names[i]
            self.cmb_fit_function.addItem(name)

        self.fnfrw = ff.FunctionsFramework()  
        self.display_curves = None
        self.residuals = None
        ## Add the data and draw them
        indmin, indmax = np.searchsorted(self.parent().blits_data.working_data['time'],(start, stop))
        self.data = cp.deepcopy(self.parent().blits_data.working_data[indmin:indmax])
        self.trace_ids = self.data.columns[self.data.columns != 'time']

        self.x_outer_limits = self.data['time'].min(), self.data['time'].max()
        self.x_limits = cp.deepcopy(self.x_outer_limits)
        self.line0, self.line1 = None, None

        self.draw_all()
        
        self.ui_ready = True
        self.on_current_index_changed(0)
   
   
    
    def get_selected_func(self):
        funcname = self.cmb_fit_function.currentText()
        return self.fn_dictionary[funcname][self.d_func]
    
    def get_selected_data(self):
        self.x_limits = sorted((self.line0.get_x(), self.line1.get_x()))
        indmin, indmax = np.searchsorted(self.data['time'], self.x_limits)
        selection = cp.deepcopy(self.data[indmin:indmax])
        data = {}
        for tid in self.trace_ids:
            x = cp.deepcopy(selection['time'])
            y = cp.deepcopy(selection[tid])
            curve = np.vstack((x,y))
            data[tid] = curve
        return data
    
    def get_constant_params(self):
        funcname = self.cmb_fit_function.currentText()
        pnames = list(self.fn_dictionary[funcname][self.d_pnames])
        c_locs = np.arange(0, len(pnames) * 3, 3) + self.hp_cons
        constants = {}
        for irow in range(self.tbl_params.rowCount()):
            cid = self.tbl_params.verticalHeaderItem(irow).text()
            if cid in self.trace_ids:     
                cpars = []
                for pname, cloc in zip(pnames, c_locs):
                    wc = self.tbl_params.item(irow, cloc)
                    if wc.checkState() == qt.Qt.Checked:
                        cpars.append(pname)
                constants[cid] = cpars
        return constants
    
    def get_linked_params(self):
        funcname = self.cmb_fit_function.currentText()
        pnames = list(self.fn_dictionary[funcname][self.d_pnames])
        links = {}
        l_locs = np.arange(0, len(pnames) * 3, 3) + self.hp_link
        for pname, lloc in zip(pnames, l_locs):
            links[pname] = []
            for irow in range(self.tbl_params.rowCount()):
                cid = self.tbl_params.verticalHeaderItem(irow).text()
                if cid in self.trace_ids:
                    w = self.tbl_params.cellWidget(irow, lloc)
                    lid = w.currentText()
                    to_append = [cid, lid]
                    for eqcls in links[pname]:
                        if cid in eqcls or lid in eqcls:
                            eqcls.extend(to_append)
                            to_append = []
                    if to_append != []:
                        links[pname].append(to_append)
        u_links = {}
        for pname in pnames:
            pulinks = []
            for eqcls in links[pname]:
                unique_linked = np.unique(np.array(eqcls))
                pulinks.append(unique_linked)
            u_links[pname] = pulinks
        return u_links    
                    
    def collect_input_for_global_fit(self):
        data = self.get_selected_data()
        func = self.get_selected_func()
        param_values = self.get_all_param_values()
        const_params = self.get_constant_params()
        links = self.get_linked_params()
        return data, func, param_values, const_params, links
              
    def on_calc(self):
        data, func, param_values, const_params, links = self.collect_input_for_global_fit()
        ff.FunctionsFramework.perform_global_curve_fit(ff.FunctionsFramework, data, func, param_values, const_params, links)
        
        self.param_values_fit, self.conf_intervals_fit, self.dw_statistic_fit = {}, {}, {}
        funcname = self.cmb_fit_function.currentText()
        f = self.fn_dictionary[funcname][self.d_func]
        self.x_limits = sorted((self.line0.get_x(), self.line1.get_x()))
        indmin, indmax = np.searchsorted(self.data['time'], self.x_limits)
        selection = cp.deepcopy(self.data[indmin:indmax])
        self.display_curves = cp.deepcopy(selection)
        self.display_curves[self.trace_ids] = np.zeros_like(selection[self.trace_ids])
        self.residuals = cp.deepcopy(selection)
        self.residuals[self.trace_ids] = np.zeros_like(selection[self.trace_ids])
        p0s = self.get_all_param_values()
        for trace in self.trace_ids:
            p = p0s[trace]
            t = selection['time']
            x = t - t.iloc[0]
            y = selection[trace]
            try:
                pfit, pcov = curve_fit(self.fnfrw.make_func(f, params=p, const={}), x, y, p0=p)
                pconf = self.fnfrw.confidence_intervals(y.shape[0], pfit, pcov, 0.95)
                self.param_values_fit[trace] = pfit
                self.conf_intervals_fit[trace] = pconf
                self.display_curves[trace] = self.fnfrw.display_curve(f, x, pfit)
                self.residuals[trace] = self.data[trace] - self.display_curves[trace]
                self.dw_statistic_fit[trace] = durbin_watson(self.residuals[trace], 0)
            except ValueError as e:
                print("VE: " + str(e))
            except RuntimeError as e:  
                print("RTE: " + str(e))
            except TypeError as e:
                print("TE: " + str(e))               
            except:
                e = sys.exc_info()[0]
                print("Generic: " + str(e))
                
        self.draw_all()
        self.prepare_results_table()   
         
    def on_current_index_changed(self, index):
        self.param_values_fit, self.conf_intervals_fit, self.dw_statistic_fit = {}, {}, {}
        if self.ui_ready:
            self.txt_function.clear()
            self.current_function = self.cmb_fit_function.currentText()
            self.txt_function.setText(self.library[self.current_function].fn_str)
            self.txt_function.adjustSize()
            self.prepare_params_table()
            self.prepare_results_table()
            
        
    def on_item_changed(self, item):
        col, row = item.column(), item.row()
        funcname = self.cmb_fit_function.currentText()
        pnames = list(self.fn_dictionary[funcname][self.d_pnames])
        if col in np.arange(0, len(pnames) * 3, 3) + self.hp_cons:
            if row == 0:
                cs = item.checkState()
                for irow in range(1, self.tbl_params.rowCount()):
                    w = self.tbl_params.item(irow, col)
                    if not w is None:
                        w.setCheckState(cs)
            else:
                cs = item.checkState()
                w_con = self.tbl_params.item(row, col)
                w_var = self.tbl_params.item(row, col - 1)
                v_con = w_con.text()
                v_var = w_var.text()
                if cs == qt.Qt.Checked: 
                    if v_con == "":
                        #w_con.setFlags(qt.Qt.ItemIsEditable)
                        w_con.setText(v_var)
                        #w_var.setFlags(w_var.flags() ^ qt.Qt.ItemIsEditable) # bitwise xor
                    else:
                        #w_var.setFlags(qt.Qt.ItemIsEditable)
                        w_var.setText("")
                        #w_con.setFlags(w_con.flags() ^ qt.Qt.ItemIsEditable) # bitwise xor
                elif cs == qt.Qt.Unchecked:
                    if v_var == "":
                        #w_var.setFlags(qt.Qt.ItemIsEditable)
                        w_var.setText(v_con)
                        w_con.setText("")
                        #w_con.setFlags(w_con.flags() ^ qt.Qt.ItemIsEditable) # bitwise xor
                self.tbl_params.resizeColumnsToContents() 
        elif col in np.arange(0, len(pnames) * 3, 3) + self.hp_link:               
            if row == 0: 
                cs = item.checkState()
                cid = self.tbl_params.verticalHeaderItem(1).text()
                for irow in range(1, self.tbl_params.rowCount()):
                    cb = self.tbl_params.cellWidget(irow, col)
                    if cs == qt.Qt.Unchecked:
                        cid = self.tbl_params.verticalHeaderItem(irow).text()
                    if not cb is None:
                        cb.setCurrentText(cid)               
                                    
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
                elif (icol - 1) % 3 in (1, ): 
                    # checkboxes under the constant header for each param
                    w = widgets.QTableWidgetItem() 
                    w.setCheckState(qt.Qt.Unchecked)
                    if irow == 0:
                        w.setText("All\nconstant")
                    if irow != 0 or icol != 0: 
                        self.tbl_params.setItem(irow, icol, w)
                elif (icol - 1) % 3 in (2, ):
                    # values of irow under linked header
                    w = widgets.QTableWidgetItem()
                    if icol != 0:
                        if irow == 0:
                            w.setText("All\nlinked")
                            w.setCheckState(qt.Qt.Unchecked)
                            self.tbl_params.setItem(irow, icol, w) 
                        else: 
                            cb = widgets.QComboBox()
                            cb.addItems(self.trace_ids)
                            cb.setCurrentText(tid)
                            self.tbl_params.setCellWidget(irow, icol, cb)
                elif (icol - 1) % 3 in (0, ) and irow != 0:
                    # initial estimate values
                    w = widgets.QTableWidgetItem()
                    npar = (icol - 1) // 3
                    w.setText('{:.2g}'.format(p0[npar]))
                    self.tbl_params.setItem(irow, icol, w)
                    
    def prepare_results_table(self):
        self.tbl_results.clear()
        labels = [self.results_table_headers[self.hr_trac],
                  self.results_table_headers[self.hr_dw],]
        
        if self.current_function != "":
            pnames = self.library[self.current_function].param_names 
            for name in pnames:
                labels.append(name + '\n' + self.results_table_headers[self.hr_pfit])
                labels.append(name + '\n' + self.results_table_headers[self.hr_conf])
                #labels.append(self.results_table_headers[self.hr_advice])
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
                w = widgets.QTableWidgetItem()
                self.tbl_results.setItem(irow, icol, w) 
                if icol == self.hr_trac: # curve colour icon in col 0
                    col = self.canvas.curve_colours[cid]
                    ic = self.parent().line_icon(col)
                    w.setIcon(ic)
                elif icol == self.hr_dw:
                    if cid in self.dw_statistic_fit:
                        w.setText('{:.4g}'.format(self.dw_statistic_fit[cid]))  
                        dw = self.dw_statistic_fit[cid]
                        if  1.0 < dw < 3.0:
                            trlcol = "green"
                        elif 0.5 < dw <= 1.0 or 3.0 <= dw < 2.5:
                            trlcol = "orange"
                        else:
                            trlcol = "red"
                        cic = self.parent().circle_icon(trlcol)
                        w.setIcon(cic)   
                else:                   
                    if cid in self.param_values_fit:
                        nparam = (icol - self.hr_pfit) // 3
                        ptype = (icol - self.hr_pfit) % 3
                        pval = self.param_values_fit[cid][nparam]
                        cintv = self.conf_intervals_fit[cid][nparam]
                        trlcol = "red"
                        rstr = "Undetermined"
                        if not np.isnan(cintv):
                            rintv = int(abs(100*cintv/pval))
                            if rintv < 20:
                                trlcol = "green"
                            elif rintv < 50:
                                trlcol = "orange"
                            else:
                                trlcol = "red"
                            if rintv <= 100:
                                rstr = '{:.1g} ({:d}%)'.format(cintv, rintv)
                            else:
                                rstr = '{:.1g} (> 100%)'.format(cintv)                                       
                        if ptype == 1:
                            w.setText(rstr)
                        if ptype == 0:
                            w.setText('{:.3g}'.format(self.param_values_fit[cid][nparam]))
                            cic = self.parent().circle_icon(trlcol)
                            w.setIcon(cic)
        self.tbl_results.resizeColumnsToContents()
        self.tbl_results.resizeRowsToContents()
                                
    def get_p0s(self):
        p0_func = self.library[self.current_function].p0_fn_ref
        x = self.data['time']
        p0s = {}
        for tid in self.trace_ids:
            y = self.data[tid]
            p0s[tid] = p0_func(x, y)
        return p0s
                               
    def get_all_param_values(self):
        """
        Returns a dictionary with trace identifiers as keys
        and a list of parameter values collected from 
        self.tbl_params (variables and constants) for each trace.
        """
        funcname = self.cmb_fit_function.currentText()
        pnames = list(self.fn_dictionary[funcname][self.d_pnames])
        p0_locs = np.arange(0, len(pnames) * 3, 3) + self.hp_p0
        c_locs = np.arange(0, len(pnames) * 3, 3) + self.hp_cons
        p0s = {}
        for irow in range(1, self.tbl_params.rowCount()):
            cid = self.tbl_params.verticalHeaderItem(irow).text()
            p0 = []
            for ploc, cloc in zip(p0_locs, c_locs):
                wp = self.tbl_params.item(irow, ploc)
                wc = self.tbl_params.item(irow, cloc)
                if wc.checkState() == qt.Qt.Checked:
                    txt = wc.text()
                    p0.append(float(txt))
                else:
                    txt = wp.text()
                    p0.append(float(txt))    
            p0s[cid] = np.array(p0) 
        return p0s    
                     
    def draw_all(self):
        if not self.data is None:
            x = self.data['time']
            y = self.data[self.trace_ids]
            self.canvas.draw_data(x, y)
            self.line0, self.line1 = None, None
            self.line0 = DraggableLine(self.canvas.data_plot.axvline(self.x_limits[0], lw=1, ls='--', color='k'), self.x_outer_limits)
            self.line1 = DraggableLine(self.canvas.data_plot.axvline(self.x_limits[1], lw=1, ls='--', color='k'), self.x_outer_limits)           
            if not self.display_curves is None:
                xd = self.display_curves['time']
                yd = self.display_curves[self.trace_ids]
                self.canvas.draw_fitted_data(xd, yd)
                ryd = self.residuals[self.trace_ids]
                self.canvas.draw_residuals(xd, ryd)
            

    def fill_library(self):                
        for name in self.fn_dictionary:
            fn_ref = self.fn_dictionary[name][self.d_func]
            p0_fn_ref = self.fn_dictionary[name][self.d_p0]
            param_names = self.fn_dictionary[name][self.d_pnames]
            fn_str = self.fn_dictionary[name][self.d_expr]
            self.library[name] = ff.ModellingFunction(name, fn_ref, p0_fn_ref, param_names, fn_str)    
            
        