'''
Created on 6 Jun 2017

@author: SchilsM
'''

#import sys
import numpy as np
import pandas as pd
import copy as cp
from statsmodels.stats.stattools import durbin_watson

from PyQt5 import QtCore as qt
from PyQt5 import QtGui as gui
from PyQt5 import QtWidgets as widgets
from blitspak.blits_mpl import MplCanvas, NavigationToolbar, DraggableLine

import functions.framework as ff
import functions.function_defs as fdefs

#import scrutinize_dialog_ui as ui

from PyQt5.uic import loadUiType

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
    params_table_headers = {head_infit: "Include\nin fit",
                            head_param_val: "Value of\n{0}",
                            head_constant: "Keep\n{0}\nconstant",
                            head_share: "Share\n{0}\nwith",
                            }
    results_table_columns = range(4)
    qual_durb_wat, qual_param_val, qual_sigma, qual_conf_lims = results_table_columns
    results_table_headers = {qual_durb_wat: "Residuals\ndistribution\n(Durbin-\nWatson)",
                             qual_param_val: "Value of\n{0}",
                             qual_sigma: "Standard \n error\non {0}",
                             qual_conf_lims: ".95\nconfidence\ninterval\nfor {0}",
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
        

#         self.fnfrw = ff.FunctionsFramework()  
        self.display_curves = None
        self.residuals = None
        
        ## Add the data and draw them
        indmin, indmax = np.searchsorted(self.parent().blits_data.working_data['time'],(start, stop))
        self.data = cp.deepcopy(self.parent().blits_data.working_data[indmin:indmax])
        self.curve_names = self.data.columns[self.data.columns != 'time'].tolist()
        # this must be generalised to cater for different data sets (with different independents)

        self.x_outer_limits = self.data['time'].min(), self.data['time'].max()
        self.x_limits = cp.deepcopy(self.x_outer_limits)
        self.line0 = DraggableLine(self.canvas.data_plot.axvline(self.x_limits[0], lw=1, ls='--', color='k'), self.x_limits)
        self.line1 = DraggableLine(self.canvas.data_plot.axvline(self.x_limits[1], lw=1, ls='--', color='k'), self.x_limits)           
        
        self.canvas.set_colours(self.curve_names)
        data = self._get_selected_data(self.curve_names)

        x_data, y_data = [], []
        for series in data:
            x = series[:-1]
            x_data.append(x)
            y = series[-1]
            y_data.append(y)
             
        self.draw_curves(self.curve_names, x_data, y_data)
        
        self.ui_ready = True
        self.on_current_index_changed(0)
        
#     def accept(self):
#         print(self.data)

    def fill_library(self):                
        for name in self.fn_dictionary:
            fn_ref = self.fn_dictionary[name][self.d_func]
            p0_fn_ref = self.fn_dictionary[name][self.d_p0]
            param_names = self.fn_dictionary[name][self.d_pnames]
            fn_str = self.fn_dictionary[name][self.d_expr]
            self.library[name] = ff.ModellingFunction(name, fn_ref, p0_fn_ref, param_names, fn_str)    
                    
    def _get_selected_series(self):
        selected = []
        for irow in range(1, self.tbl_params.rowCount()):
            cname = self.tbl_params.verticalHeaderItem(irow).text()
            if cname in self.curve_names: # superfluous, but just in case...
                in_fit = self.tbl_params.item(irow, 0)
                if in_fit.checkState() == qt.Qt.Checked:
                    selected.append(cname)
        return selected
            
   
    def _get_selected_data(self, series_names):
        self.x_limits = sorted((self.line0.get_x(), self.line1.get_x()))
        indmin, indmax = np.searchsorted(self.data['time'], self.x_limits)
        selection = cp.deepcopy(self.data[indmin:indmax])
        data = []
        for tid in series_names:
            x = cp.deepcopy(selection['time'])
            y = cp.deepcopy(selection[tid])
            curve = np.vstack((x,y))
            if len(data) == 0:
                data = [curve]
            else:
                data.append(curve)
        return data

    def _get_all_param_values(self, series_names):
        """
        Returns an (n_curves, n_params)-shaped array (with rows and columns 
        parallel to self.curve_names and self.fn_dictionary[fn][self.d_pnames], 
        respectively) with values for each parameter in each curve).  
        """
#         ncol_per_param = 2
#         if self.chk_global.checkState() == qt.Qt.Checked:
#             ncol_per_param = 3
        ncol_per_param = len(self.params_table_headers) - 2 #3
        if self.chk_global.checkState() == qt.Qt.Checked:
            ncol_per_param += 1                 

        funcname = self.cmb_fit_function.currentText()
        param_names = list(self.fn_dictionary[funcname][self.d_pnames])
        
        p_locs = np.arange(0, len(param_names) * ncol_per_param, ncol_per_param) + self.head_param_val
        p_vals = np.zeros((len(self.curve_names), len(param_names)))
        
        for irow in range(1, self.tbl_params.rowCount()):
            cname = self.tbl_params.verticalHeaderItem(irow).text()
            if cname in self.curve_names:
                cind = self.curve_names.index(cname)
                pval = []
                for ploc in p_locs:
                    txt = self.tbl_params.item(irow, ploc).text()
                    if len(pval) == 0:
                        pval = [float(txt)]
                    else:
                        pval.append(float(txt))
                p_vals[cind] = np.array(pval)
                
        selected = [self.curve_names.index(name) for name in series_names] 
        return p_vals[selected]   
    
    def _get_constant_params(self, series_names):
        """
        Returns an (n_curves, n_params)-shaped array of Boolean values 
        (with rows and columns parallel to self.curve_names and 
        self.fn_dictionary[fn][self.d_pnames], respectively); if True, 
        parameter values is constant, if False, parameter value is variable.
        """
#         ncol_per_param = 2
#         if self.chk_global.checkState() == qt.Qt.Checked:
#             ncol_per_param = 3
        ncol_per_param = len(self.params_table_headers) - 2 #3
        if self.chk_global.checkState() == qt.Qt.Checked:
            ncol_per_param += 1                 

        funcname = self.cmb_fit_function.currentText()
        param_names = list(self.fn_dictionary[funcname][self.d_pnames])
        
        cnst_locs = np.arange(0, len(param_names) * ncol_per_param, ncol_per_param) + self.head_constant
        const_params = np.zeros((len(self.curve_names), len(param_names)), dtype = bool)
        for irow in range(1, self.tbl_params.rowCount()):
            cname = self.tbl_params.verticalHeaderItem(irow).text()
            if cname in self.curve_names: 
                cind = self.curve_names.index(cname)
                for pname, cloc in zip(param_names, cnst_locs):
                    pind = param_names.index(pname)
                    const_params[cind, pind] = self.tbl_params.item(irow, cloc).checkState() == qt.Qt.Checked
                    
        selected = [self.curve_names.index(name) for name in series_names]    
        return const_params[selected]
    
    def _get_linked_params(self, series_names):
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
#         ncol_per_param = 2
#         if self.chk_global.checkState() == qt.Qt.Checked:
#             ncol_per_param = 3
        ncol_per_param = len(self.params_table_headers) - 2 #3
        if self.chk_global.checkState() == qt.Qt.Checked:
            ncol_per_param += 1                 

        funcname = self.cmb_fit_function.currentText()
        param_names = list(self.fn_dictionary[funcname][self.d_pnames])
        
        nparams = len(param_names)
        ncurves = len(self.curve_names) 
        links = np.arange(nparams * ncurves, dtype=int)
        links = np.reshape(links, (nparams, ncurves))
        
        if self.chk_global.checkState() == qt.Qt.Checked:
            l_locs = np.arange(0, nparams * ncol_per_param, ncol_per_param) + self.head_share
            pcount, indpcount = 0, 0
            for lloc in l_locs:
                # Find all connections (reflexive, symmetrical, transitive graph)
                mlinks = np.identity(ncurves, dtype=int) # Make matrix reflexive
                for irow in range(self.tbl_params.rowCount()):
                    cname = self.tbl_params.verticalHeaderItem(irow).text()
                    if cname in self.curve_names:
                        linked = self.tbl_params.cellWidget(irow, lloc).currentText()
                        cind = self.curve_names.index(cname)
                        lind = self.curve_names.index(linked)
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
        
        selected = [self.curve_names.index(name) for name in series_names]        
        return links.transpose()[selected]
                                
    def on_calc(self):
        funcname = self.cmb_fit_function.currentText()
        func = self.fn_dictionary[funcname][self.d_func]
        selected_series = self._get_selected_series()
        data = self._get_selected_data(selected_series) 
        param_values = self._get_all_param_values(selected_series)
        const_params = self._get_constant_params(selected_series)
        links = self._get_linked_params(selected_series)
        fitted_params = cp.deepcopy(param_values)
        sigmas = np.empty_like(fitted_params)
        confidence_intervals = np.empty_like(fitted_params)
        tol = None
        
        self.set_tbl_qual_values() # no values; sets all cell contents to ""
                
        ffw = ff.FunctionsFramework()
        if self.chk_global.checkState() == qt.Qt.Checked:
            results = ffw.perform_global_curve_fit(data, func, param_values, const_params, links)
            fitted_params = results[0]
            sigmas = results[1]
            confidence_intervals = results[2]
            tol = results[3]
        else:
            tol = []
            n = 0
            for d, p, c, l in zip(data, param_values, const_params, links):
                d = [d, ]
                p = np.reshape(p, (1, p.shape[0]))
                c = np.reshape(c, (1, c.shape[0]))
                l = np.reshape(l, (1, l.shape[0]))
                results = ffw.perform_global_curve_fit(d, func, p, c, l)
                fitted_params[n] = results[0]
                sigmas[n] = results[1]
                confidence_intervals[n] = results[2]
                tol.append(results[3])
                n += 1
        
        
        fitted_curves = cp.deepcopy(data) # data is a list of arrays in which [:-1] are x-values and [-1] is y
        x_data, y_data, y_fit_data, y_res_data = [], [], [], []
        fitted_param_dict, sigma_dict, conf_intv_dict, dw_stat_dict = {}, {}, {}, {}
        
        for sid, series, params, sigma, conf_intv in zip(selected_series, 
                                                  fitted_curves, 
                                                  fitted_params, 
                                                  sigmas,
                                                  confidence_intervals, 
                                                  ):
            x = series[:-1]
            y = series[-1]
            y_fit = func(x, params)
            y_res = y - y_fit
            dw_stat = durbin_watson(y_res, 0)

            x_data.append(x)
            y_data.append(y)
            y_fit_data.append(y_fit)
            y_res_data.append(y_res)
            fitted_param_dict[sid] = params
            sigma_dict[sid] = sigma
            conf_intv_dict[sid] = conf_intv
            dw_stat_dict[sid] = dw_stat

        self.draw_curves(selected_series, x_data, y_data, y_fit_data, y_res_data)
        self.set_tbl_param_values(fitted_param_dict)
        self.set_tbl_qual_values(fitted_param_dict, sigma_dict, conf_intv_dict, dw_stat_dict)
        
            
    def draw_curves(self, series_ids, x_data, y_data, y_fit_data=[], y_residual_data=[]):
        self.canvas.clear_plots()
        self.line0 = DraggableLine(self.canvas.data_plot.axvline(self.x_limits[0], lw=1, ls='--', color='k'), self.x_limits)
        self.line1 = DraggableLine(self.canvas.data_plot.axvline(self.x_limits[1], lw=1, ls='--', color='k'), self.x_limits)           

        for sid, x, y in zip(series_ids, x_data, y_data):            
            self.canvas.draw_series(sid, x[0], y)
        if y_fit_data != []:
            for sid, x, y_fit in zip(series_ids, x_data, y_fit_data):            
                self.canvas.draw_series_fit(sid, x[0], y_fit)
        if y_residual_data != []:
            for sid, x, y_res in zip(series_ids, x_data, y_residual_data):            
                self.canvas.draw_series_residuals(sid, x[0], y_res)
            
            

    def on_toggle_global(self):
        self.prepare_params_table()
         
    def on_current_index_changed(self, index):
        if self.ui_ready:
            self.txt_function.clear()
            self.current_function = self.cmb_fit_function.currentText()
            self.txt_function.setText(self.library[self.current_function].fn_str)
            self.txt_function.adjustSize()
            self.prepare_params_table()
            self.prepare_results_table()
            
    def on_item_changed(self, item):
        ncol_per_param = len(self.params_table_headers) - 2 #3
        if self.chk_global.checkState() == qt.Qt.Checked:
            ncol_per_param += 1                 
                      
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
        # Set up table for global or non-global fit
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
                labels.append(self.params_table_headers[self.head_param_val].format(name))
                labels.append(self.params_table_headers[self.head_constant].format(name))
                if global_fit:
                    labels.append(self.params_table_headers[self.head_share].format(name))
        self.tbl_params.setColumnCount(len(labels))
        self.tbl_params.setHorizontalHeaderLabels(labels)
        ncol_per_param = len(self.params_table_headers) - 2
        if global_fit:
            ncol_per_param += 1 #3
        
        # Set vertical header (series names + colour icons)
        if len(self.curve_names) != 0:
            labels = ['All',]
            labels.extend(self.curve_names)
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
        
    def set_tbl_param_values(self, param_val_dict):
        ncol_per_param = len(self.params_table_headers) - 2 #3
        if self.chk_global.checkState() == qt.Qt.Checked:
            ncol_per_param += 1                 
        for irow in range(self.tbl_params.rowCount()):
            sid = self.tbl_params.verticalHeaderItem(irow).text()
            if sid in param_val_dict:
                params_for_series = param_val_dict[sid] #[self.curve_names[irow-1]]
                for icol in range(self.tbl_params.columnCount()):
                    if (icol - 1) % ncol_per_param in (0, ) and irow != 0:
                        w = self.tbl_params.item(irow, icol)
                        npar = (icol - 1) // ncol_per_param
                        w.setText('{:.2g}'.format(params_for_series[npar]))
        self.tbl_params.resizeColumnsToContents()
        self.tbl_params.resizeRowsToContents()
                                
    def get_p0s(self):
        p0_func = self.library[self.current_function].p0_fn_ref
        x = self.data['time']
        param_values = {}
        for tid in self.curve_names:
            y = self.data[tid]
            param_values[tid] = p0_func(x, y)
        return param_values
                               
    def prepare_results_table(self):
        # Clear the table of all data
        self.tbl_results.clear()
        # Set the horizontal header for the current function
        labels = [self.results_table_headers[self.qual_durb_wat]]
        if self.current_function != "":
            param_names = self.library[self.current_function].param_names 
            for name in param_names:
                labels.append(self.results_table_headers[self.qual_param_val].format(name))
                labels.append(self.results_table_headers[self.qual_sigma].format(name))
                labels.append(self.results_table_headers[self.qual_conf_lims].format(name))
        self.tbl_results.setColumnCount(len(labels))
        self.tbl_results.setHorizontalHeaderLabels(labels)
        
        # Set vertical header (series names + colour icons)
        if len(self.curve_names) != 0:
            labels = self.curve_names
            self.tbl_results.setRowCount(len(labels))
            row = 0
            for lbl in labels:
                vhw = widgets.QTableWidgetItem()
                vhw.setText(lbl)
                vhw.setTextAlignment(qt.Qt.AlignRight)
                self.tbl_results.setVerticalHeaderItem(row, vhw)
                clr = self.canvas.curve_colours[lbl]
                ic = self.parent().line_icon(clr)
                vhw.setIcon(ic)
                row += 1
                
        for icol in range(self.tbl_results.columnCount()):
            for irow in range(self.tbl_results.rowCount()):
                w = widgets.QTableWidgetItem()
                self.tbl_results.setItem(irow, icol, w)
        
        self.tbl_results.resizeColumnsToContents()
        self.tbl_results.resizeRowsToContents()
                
    def set_tbl_qual_values(self, param_val_dict={}, sigma_dict={}, conf_intv_dict={}, durbwat_stat_dict={}):
        ncol_per_param = len(self.params_table_headers) - 1 #3
        for irow in range(self.tbl_results.rowCount()):
            sid = self.tbl_results.verticalHeaderItem(irow).text()
            icol = 0
            wd = self.tbl_results.item(irow, icol)
            wd.setText("")
            if sid in durbwat_stat_dict:
                dw = durbwat_stat_dict[sid]
                wd.setText('{:.3g}'.format(dw)) 
                if  1.0 < dw < 3.0:
                    trlcol = "green"
                elif 0.5 < dw <= 1.0 or 3.0 <= dw < 2.5:
                    trlcol = "orange"
                else:
                    trlcol = "red"
                cic = self.parent().circle_icon(trlcol)
                wd.setIcon(cic)   

            for icol in range(1, self.tbl_results.columnCount(), ncol_per_param):
                if (icol-1) % ncol_per_param in (0, ):
                    npar = (icol-1) // ncol_per_param
                    wv = self.tbl_results.item(irow, icol)
                    wv.setText("")
                    ws = self.tbl_results.item(irow, icol+1)
                    ws.setText("")
                    wc = self.tbl_results.item(irow, icol+2)
                    wc.setText("")
                    v, c = 1, 0
                    if sid in param_val_dict:
                        v = param_val_dict[sid][npar]
                        wv.setText('{:.2g}'.format(v))
                    if sid in sigma_dict:
                        s = sigma_dict[sid][npar]
                        ws.setText('{:.2g}'.format(s))
                    if sid in conf_intv_dict:
                        c = conf_intv_dict[sid][npar]
                        wc.setText('{:.2g}'.format(abs(c)))
                        
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
                    

#     def draw_all(self):
#         if not self.data is None:
#             x = self.data['time']
#             y = self.data[self.curve_names]
#             self.canvas.draw_data(x, y)
#             self.line0, self.line1 = None, None
#             self.line0 = DraggableLine(self.canvas.data_plot.axvline(self.x_limits[0], lw=1, ls='--', color='k'), self.x_outer_limits)
#             self.line1 = DraggableLine(self.canvas.data_plot.axvline(self.x_limits[1], lw=1, ls='--', color='k'), self.x_outer_limits)           
#             if not self.display_curves is None:
#                 xd = self.display_curves['time']
#                 yd = self.display_curves[self.curve_names]
#                 self.canvas.draw_fitted_data(xd, yd)
#                 ryd = self.residuals[self.curve_names]
#                 self.canvas.draw_residuals(xd, ryd)

