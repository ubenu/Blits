'''
Created on 6 Jun 2017

@author: SchilsM
'''

#import sys
import numpy as np
import pandas as pd
import copy as cp
#from scipy.optimize import curve_fit
#from statsmodels.stats.stattools import durbin_watson

from PyQt5 import QtCore as qt
from PyQt5 import QtGui as gui
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
    
    # Function selection is a kind of stub: needs to go via dialog
    # and offer possibility for users to create their own function
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
                                 "Series average"),
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
                                            "a0 + a1.exp(-x.k1) + a2.exp(-x.k2)"), 
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
    head_infit, head_param_val, head_constant, head_share = params_table_columns
    params_table_headers = {head_infit: "Fit\nseries",
                            head_param_val: "Parameter\nvalue",
                            head_constant: "Keep\nvalue\nconstant",
                            head_share: "Share\nvalue\nwith",
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
        self.mpl_layout.setAlignment(qt.Qt.AlignHCenter)
        
        ## Connect signals to slots
        self.cmb_fit_function.currentIndexChanged.connect(self.on_current_index_changed)
        self.tbl_params.itemChanged.connect(self.on_item_changed)
        self.btn_calc.clicked.connect(self.on_calc)
        self.chk_global.clicked.connect(self.on_toggle_global)
        
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
        self.curve_names = self.data.columns[self.data.columns != 'time']

        self.x_outer_limits = self.data['time'].min(), self.data['time'].max()
        self.x_limits = cp.deepcopy(self.x_outer_limits)
        self.line0, self.line1 = None, None

        self.draw_all()
        
        self.ui_ready = True
        self.on_current_index_changed(0)
   
    def _get_selected_data(self):
        self.x_limits = sorted((self.line0.get_x(), self.line1.get_x()))
        indmin, indmax = np.searchsorted(self.data['time'], self.x_limits)
        selection = cp.deepcopy(self.data[indmin:indmax])
        data = []
        for tid in self.curve_names:
            x = cp.deepcopy(selection['time'])
            y = cp.deepcopy(selection[tid])
            curve = np.vstack((x,y))
            if len(data) == 0:
                data = [curve]
            else:
                data.append(curve)
        return data

    def _get_all_param_values(self):
        """
        Returns an (n_curves, n_params)-shaped array (with rows and columns 
        parallel to self.curve_names and self.fn_dictionary[fn][self.d_pnames], 
        respectively) with values for each parameter in each curve).  
        """
        ncol_per_param = 2
        if self.chk_global.checkState() == qt.Qt.Checked:
            ncol_per_param = 3

        funcname = self.cmb_fit_function.currentText()
        param_names = list(self.fn_dictionary[funcname][self.d_pnames])
        
        p_locs = np.arange(0, len(param_names) * ncol_per_param, ncol_per_param) + self.head_param_val
        p_vals = np.zeros((len(self.curve_names), len(param_names)))
        
        for irow in range(1, self.tbl_params.rowCount()):
            cname = self.tbl_params.verticalHeaderItem(irow).text()
            if cname in self.curve_names:
                cind = self.curve_names.get_loc(cname)
                pval = []
                for ploc in p_locs:
                    txt = self.tbl_params.item(irow, ploc).text()
                    if len(pval) == 0:
                        pval = [float(txt)]
                    else:
                        pval.append(float(txt))
                p_vals[cind] = np.array(pval)
        return p_vals    
    
    def _get_constant_params(self):
        """
        Returns an (n_curves, n_params)-shaped array of Boolean values 
        (with rows and columns parallel to self.curve_names and 
        self.fn_dictionary[fn][self.d_pnames], respectively); if True, 
        parameter values is constant, if False, parameter value is variable.
        """
        ncol_per_param = 2
        if self.chk_global.checkState() == qt.Qt.Checked:
            ncol_per_param = 3

        funcname = self.cmb_fit_function.currentText()
        param_names = list(self.fn_dictionary[funcname][self.d_pnames])
        
        cnst_locs = np.arange(0, len(param_names) * ncol_per_param, ncol_per_param) + self.head_constant
        const_params = np.zeros((self.curve_names.shape[0], len(param_names)), dtype = bool)
        for irow in range(1, self.tbl_params.rowCount()):
            cname = self.tbl_params.verticalHeaderItem(irow).text()
            if cname in self.curve_names: 
                cind = self.curve_names.get_loc(cname)
                for pname, cloc in zip(param_names, cnst_locs):
                    pind = param_names.index(pname)
                    const_params[cind, pind] = self.tbl_params.item(irow, cloc).checkState() == qt.Qt.Checked
        return const_params
    
    def _get_linked_params(self):
        """
        Returns an (n_curves, n_params)-shaped array (with rows and columns parallel 
        to self.curve_names and self.fn_dictionary[fn][self.d_pnames], respectively)
        of integers, in which linked parameters are grouped by their values.
        Example for 4 curves and 3 parameters:
              p0    p1    p2
        c0    0     2     3
        c1    0     2     4
        c2    1     2     5
        c3    1     2     6
        indicates that parameter p0 is assumed to have the same value in 
        curves c0 and c1, and in curves c2 and c3 (a different value), 
        and that the value for p1 is the same in all curves, whereas
        the value of p2 is different for all curves. 
        """
        ncol_per_param = 2
        if self.chk_global.checkState() == qt.Qt.Checked:
            ncol_per_param = 3

        funcname = self.cmb_fit_function.currentText()
        param_names = list(self.fn_dictionary[funcname][self.d_pnames])
        
        nparams = len(param_names) # param_names is a python list
        ncurves = self.curve_names.shape[0]  # self.curve_names is a pandas Index
        l_locs = np.arange(0, nparams * ncol_per_param, ncol_per_param) + self.head_share
        links = np.empty((nparams, ncurves), dtype=int)
        pcount, indpcount = 0, 0
        for lloc in l_locs:
            # Find all connections (reflexive, symmetrical, transitive graph)
            mlinks = np.identity(ncurves, dtype=int) # Make matrix reflexive
            for irow in range(self.tbl_params.rowCount()):
                cname = self.tbl_params.verticalHeaderItem(irow).text()
                if cname in self.curve_names:
                    linked = self.tbl_params.cellWidget(irow, lloc).currentText()
                    cind = self.curve_names.get_loc(cname)
                    lind = self.curve_names.get_loc(linked)
                    mlinks[cind, lind] = 1
                    mlinks[lind, cind] = 1 # Make matrix symmetrical
            # Warshall-Floyd to make matrix transitive 
            for k in range(ncurves):
                for i in range(ncurves):
                    for j in range(ncurves):
                        mlinks[i, j] = mlinks[i, j] or (mlinks[i, k] == 1 and mlinks[k, j] == 1)
            # Find the equivalence classes for this parameter 
            scrap = np.ones((ncurves,), dtype=bool)
            eq_classes = []
            for k in range(ncurves):
                if scrap[k]:
                    ec = np.where(mlinks[k] == 1)
                    eq_classes.append(ec[0])
                    scrap[ec] = False
            # Number the individual equivalence classes
            ind_params = np.empty_like(self.curve_names, dtype=int)
            for i in eq_classes:
                ind_params[i] = indpcount
                indpcount += 1
            links[pcount] = ind_params
            pcount += 1
        return links.transpose()
                                
    def on_calc(self):
        funcname = self.cmb_fit_function.currentText()
        func = self.fn_dictionary[funcname][self.d_func]
        data = self._get_selected_data()
        param_values = self._get_all_param_values()
        const_params = self._get_constant_params()
        links = None
        if self.chk_global.checkState() == qt.Qt.Checked:
            links = self._get_linked_params()
        curve_names = self.curve_names.tolist()
        header = ['time']
        header.extend(curve_names)
#        param_names = list(self.fn_dictionary[funcname][self.d_pnames])
        
        ffw = ff.FunctionsFramework()
        fitted_params = ffw.perform_fit(data, func, param_values, const_params, links)
        
        fitted_curves = cp.deepcopy(data)
        d_curves = None
        r_curves = None
        for curve, params in zip(fitted_curves, fitted_params):
            x = curve[:-1]
            y = curve[-1]
            y_fit = func(x, params)
            y_res = y - y_fit
            if d_curves is None:
                d_curves = np.vstack((x[0], y_fit))
                r_curves = np.vstack((x[0], y_res))
            else:
                d_curves = np.vstack((d_curves, y_fit))
                r_curves = np.vstack((r_curves, y_res))
        self.display_curves = pd.DataFrame(d_curves.transpose(), columns=header)
        self.residuals = pd.DataFrame(r_curves.transpose(), columns=header)
        self.draw_all()

    def on_toggle_global(self):
        self.prepare_params_table()
         
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
        ncol_per_param = 2
        if self.chk_global.checkState() == qt.Qt.Checked:
            ncol_per_param = 3
                      
        funcname = self.cmb_fit_function.currentText()
        param_names = list(self.fn_dictionary[funcname][self.d_pnames])
        
        col, row = item.column(), item.row()
        if col == 0 or col in np.arange(0, len(param_names) * ncol_per_param, ncol_per_param) + self.head_constant:
            if row == 0:
                cs = item.checkState()
                for irow in range(1, self.tbl_params.rowCount()):
                    w = self.tbl_params.item(irow, col)
                    if not w is None:
                        w.setCheckState(cs)
        elif col in np.arange(0, len(param_names) * ncol_per_param, ncol_per_param) + self.head_share:               
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
        # Table must be set up for global or non-global fit
        global_fit = False
        if self.chk_global.checkState() == qt.Qt.Checked:
            global_fit = True
        # Clear the table of all data
        self.tbl_params.clear()

        # Set the horizontal header for the current function
        labels = [self.params_table_headers[self.head_infit], ]
        if self.current_function != "":
            param_names = self.library[self.current_function].param_names 
            for name in param_names:
                labels.append(name)
                labels.append(self.params_table_headers[self.head_constant])
                if global_fit:
                    labels.append(self.params_table_headers[self.head_share])
        self.tbl_params.setColumnCount(len(labels))
        self.tbl_params.setHorizontalHeaderLabels(labels)
        ncol_per_param = 2
        if global_fit:
            ncol_per_param = 3
        
        # Set vertical header (series names + colour icons)
        if len(self.curve_names) != 0:
            labels = ['All',]
            labels.extend(self.curve_names.tolist())
            self.tbl_params.setRowCount(len(labels))
            row = 0
            for lbl in labels:
                vhw = widgets.QTableWidgetItem()
                vhw.setText(lbl)
                vhw.setTextAlignment(qt.Qt.AlignRight)
                self.tbl_params.setVerticalHeaderItem(row, vhw)
                if row > 0:
                    clr = self.canvas.curve_colours[lbl]
                    ic = self.parent().line_icon(clr)
                    vhw.setIcon(ic)
                row += 1

        # Set horizontal header
        for irow in range(self.tbl_params.rowCount()):
            for icol in range(self.tbl_params.columnCount()):
                # checkboxes under the in-fit header for each param
                if icol == 0:
                    w = widgets.QTableWidgetItem()
                    w.setCheckState(qt.Qt.Checked)
                    self.tbl_params.setItem(irow, icol, w)
                elif (icol - 1) % ncol_per_param in (1, ): 
                    # checkboxes under the constant header for each param
                    if irow != 0 or icol != 0: 
                        w = widgets.QTableWidgetItem()
                        w.setCheckState(qt.Qt.Unchecked)
                        self.tbl_params.setItem(irow, icol, w)
                elif global_fit and (icol - 1) % ncol_per_param in (2, ):
                    # values of irow under linked header
                    if icol != 0:
                        if irow == 0:
                            w = widgets.QTableWidgetItem()
                            w.setCheckState(qt.Qt.Unchecked)
                            self.tbl_params.setItem(irow, icol, w) 
                        else: 
                            cb = widgets.QComboBox()
                            cb.addItems(self.curve_names)
                            cb.setCurrentText(self.curve_names[irow-1])
                            self.tbl_params.setCellWidget(irow, icol, cb)
                elif (icol - 1) % ncol_per_param in (0, ) and irow != 0:
                    # initial estimate values
                    w = widgets.QTableWidgetItem()
                    self.tbl_params.setItem(irow, icol, w)
        p0s = self.get_p0s()
        self.set_tbl_param_values(p0s)
        self.tbl_params.resizeColumnsToContents()
        self.tbl_params.resizeRowsToContents()
        
    def set_tbl_param_values(self, param_val_dict):
        ncol_per_param = 2
        if self.chk_global.checkState() == qt.Qt.Checked:
            ncol_per_param = 3  
               
        for irow in range(self.tbl_params.rowCount()):
            params_for_series = param_val_dict[self.curve_names[irow-1]]
            for icol in range(self.tbl_params.columnCount()):
                if (icol - 1) % ncol_per_param in (0, ) and irow != 0:
                    w = self.tbl_params.item(irow, icol)
                    npar = (icol - 1) // ncol_per_param
                    w.setText('{:.2g}'.format(params_for_series[npar]))
                                
    def get_p0s(self):
        p0_func = self.library[self.current_function].p0_fn_ref
        x = self.data['time']
        param_values = {}
        for tid in self.curve_names:
            y = self.data[tid]
            param_values[tid] = p0_func(x, y)
        return param_values
                               
                     
    def draw_all(self):
        if not self.data is None:
            x = self.data['time']
            y = self.data[self.curve_names]
            self.canvas.draw_data(x, y)
            self.line0, self.line1 = None, None
            self.line0 = DraggableLine(self.canvas.data_plot.axvline(self.x_limits[0], lw=1, ls='--', color='k'), self.x_outer_limits)
            self.line1 = DraggableLine(self.canvas.data_plot.axvline(self.x_limits[1], lw=1, ls='--', color='k'), self.x_outer_limits)           
            if not self.display_curves is None:
                xd = self.display_curves['time']
                yd = self.display_curves[self.curve_names]
                self.canvas.draw_fitted_data(xd, yd)
                ryd = self.residuals[self.curve_names]
                self.canvas.draw_residuals(xd, ryd)
            

    def fill_library(self):                
        for name in self.fn_dictionary:
            fn_ref = self.fn_dictionary[name][self.d_func]
            p0_fn_ref = self.fn_dictionary[name][self.d_p0]
            param_names = self.fn_dictionary[name][self.d_pnames]
            fn_str = self.fn_dictionary[name][self.d_expr]
            self.library[name] = ff.ModellingFunction(name, fn_ref, p0_fn_ref, param_names, fn_str)    
            
    def prepare_results_table(self):
        self.tbl_results.clear()
        labels = [self.results_table_headers[self.hr_trac],
                  self.results_table_headers[self.hr_dw],]
        
        if self.current_function != "":
            param_names = self.library[self.current_function].param_names 
            for name in param_names:
                labels.append(name + '\n' + self.results_table_headers[self.hr_pfit])
                labels.append(name + '\n' + self.results_table_headers[self.hr_conf])
                #labels.append(self.results_table_headers[self.hr_advice])
        self.tbl_results.setColumnCount(len(labels))
        self.tbl_results.setHorizontalHeaderLabels(labels)
        if len(self.curve_names) != 0:
            labels = []
            labels.extend(self.curve_names.tolist())
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
        
#     def centred_checkbox(self, checked=False):
#         w = widgets.QWidget() 
#         c = widgets.QCheckBox()
#         l = widgets.QVBoxLayout(w)
#         c.setCheckState(qt.Qt.Unchecked)
#         if checked:
#             c.setCheckState(qt.Qt.Checked)
#         c.stateChanged.connect(self.on_checkstate_changed)
#         l.addWidget(c)
#         l.setAlignment(qt.Qt.AlignCenter)
#         l.setContentsMargins(0, 0, 0, 0)
#         w.setLayout(l)
#         return w                    
                    
      
        
###    NON-GLOBAL FITTING ROUTINE
##         self.param_values_fit, self.conf_intervals_fit, self.dw_statistic_fit = {}, {}, {}
##         funcname = self.cmb_fit_function.currentText()
##         f = self.fn_dictionary[funcname][self.d_func]
##         self.x_limits = sorted((self.line0.get_x(), self.line1.get_x()))
##         indmin, indmax = np.searchsorted(self.data['time'], self.x_limits)
##         selection = cp.deepcopy(self.data[indmin:indmax])
#         self.display_curves = cp.deepcopy(selection)
#         self.display_curves[self.curve_names] = np.zeros_like(selection[self.curve_names])
#         self.residuals = cp.deepcopy(selection)
#         self.residuals[self.curve_names] = np.zeros_like(selection[self.curve_names])
#         param_values = self.get_all_param_values()
#         for trace in self.curve_names:
#             p = param_values[trace]
#             t = selection['time']
#             x = t - t.iloc[0]
#             y = selection[trace]
#             try:
#                 pfit, pcov = curve_fit(self.fnfrw.make_func(f, params=p, const={}), x, y, p0=p)
#                 pconf = self.fnfrw.confidence_intervals(y.shape[0], pfit, pcov, 0.95)
#                 self.param_values_fit[trace] = pfit
#                 self.conf_intervals_fit[trace] = pconf
#                 self.display_curves[trace] = self.fnfrw.display_curve(f, x, pfit)
#                 self.residuals[trace] = self.data[trace] - self.display_curves[trace]
#                 self.dw_statistic_fit[trace] = durbin_watson(self.residuals[trace], 0)
#             except ValueError as e:
#                 print("VE: " + str(e))
#             except RuntimeError as e:  
#                 print("RTE: " + str(e))
#             except TypeError as e:
#                 print("TE: " + str(e))               
#             except:
#                 e = sys.exc_info()[0]
#                 print("Generic: " + str(e))
                
#         self.draw_all()
#         self.prepare_results_table()   


