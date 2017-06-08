'''
Created on 6 Jun 2017

@author: SchilsM
'''
from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets
from blitspak.blits_mpl import MplCanvas, NavigationToolbar

import functions.framework as ff
import functions.function_defs as fd

#import scrutinize_dialog_ui as ui

from PyQt5.uic import loadUiType
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

    fn_dictionary = {"Average": (fd.fn_average, 
                                 ('a',), 
                                 "average(trace)"),
                     "Straight line": (fd.fn_straight_line, 
                                       ('a', 'b'), 
                                       "a + b.x"), 
                     "Single exponential": (fd.fn_1exp, 
                                            ('a0', 'a1', 'k1'), 
                                            "a0 + a1.exp(-x.k1)"),
                     "Double exponential": (fd.fn_2exp, 
                                            ('a0', 'a1', 'k1', 'a2', 'k2'), 
                                            "a0 + a1.exp(-x.k1) + a2..exp(-x.k2)"), 
                     "Triple exponential": (fd.fn_3exp, 
                                            ('a0', 'a1', 'k1', 'a2', 'k2', 'a3', 'k3'), 
                                            "a0 + a1.exp(-x.k1) + a2.exp(-x.k2) + a3.exp(-x.k3)"),
                     "Michaelis-Menten equation": (fd.fn_mich_ment, 
                                                   ('km', 'vmax'), 
                                                   "vmax . x / (km + x)"),
                     "Competitive inhibition equation": (fd.fn_comp_inhibition, 
                                                         ('km', 'ki', 'vmax'), 
                                                         "vmax . x[0] / (km . (1.0 + x[1] / ki) + x[0])"), 
                     "Hill equation": (fd.fn_hill, 
                                       ('ymax', 'xhalf', 'h'), 
                                       "ymax / ((xhalf/x)^h + 1 )"),
                     }
    
    params_table_columns = range(4)
    hp_trac, hp_p0, hp_cons, hp_link = params_table_columns
    params_table_headers = {hp_trac: "Trace",
                            hp_p0: "Param",
                            hp_cons: "Const",
                            hp_link: "Linked",
                            }
    results_table_columns = range(4)
    hr_trac, hr_p0, hr_conf, hr_advice = results_table_columns
    results_table_headers = {hr_trac: "Trace",
                             hr_p0: "",
                             hr_conf: "Confidence\interval (%)",
                             hr_advice: "",
                             }

    def __init__(self, parent, trace_segments):
        '''
        Constructor
        '''
        super(widgets.QDialog, self).__init__(parent)
        self.setupUi(self)

        self.cmb_fit_function.currentIndexChanged.connect(self.on_current_index_changed)
        self.tbl_params.itemChanged.connect(self.on_item_changed)
        
        self.library = {}
        self.fill_library()
        self.current_function = ""
        self.trace_ids = []
        self.cmb_fit_function.setSizeAdjustPolicy(widgets.QComboBox.AdjustToContents)
        self.populate_lib_combo()
        self.cmb_fit_function.setCurrentIndex(0)
        
        self.canvas = MplCanvas(self.mpl_window)
        self.mpl_layout.addWidget(self.canvas)
        self.plot_toolbar = NavigationToolbar(self.canvas, self.mpl_window)
        self.mpl_layout.addWidget(self.plot_toolbar)
        
        self.data = trace_segments
        self.trace_ids = self.data.columns[self.data.columns != 'time']
        self.draw_data()
        self.on_current_index_changed(self.cmb_fit_function.currentIndex())
        
    def on_current_index_changed(self, index):
        self.txt_function.clear()
        self.current_function = self.cmb_fit_function.currentText()
        self.txt_function.setText(self.library[self.current_function].fn_str)
        self.txt_function.adjustSize()
        self.prepare_params_table()
        
    def on_item_changed(self, item):
        col, row = item.column(), item.row()
        if row == 0 and (col - 1) % 3 == 2:
            cs = item.checkState()
            print(cs)
            for irow in range(1,self.tbl_params.rowCount()):
                w = self.tbl_params.item(irow, col)
                if not w is None:
                    w.setCheckState(cs)
                               
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
            
        for irow in range(self.tbl_params.rowCount()):
            for icol in range(self.tbl_params.columnCount()):
                if icol == 0 and irow != 0:
                    w = widgets.QTableWidgetItem()
                    cid = self.tbl_params.verticalHeaderItem(irow).text()
                    col = self.canvas.curve_colours[cid]
                    ic = self.parent().line_icon(col)
                    w.setIcon(ic)
                    self.tbl_params.setItem(irow, icol, w)
                elif (icol - 1) % 3 in (1, 2):
                    w = widgets.QTableWidgetItem() #widgets.QWidget()
                    w.setCheckState(qt.Qt.Unchecked)
                    if irow != 0 or (icol - 1) % 3 == 2 and icol != 0:
                        self.tbl_params.setItem(irow, icol, w)
                elif (icol - 1) % 3 in (0, ) and irow != 0:
                    w = widgets.QTableWidgetItem()
                    w.setText(str(1.0))
                    self.tbl_params.setItem(irow, icol, w)
                    
    def on_check_state_changed(self):
        print("Hello")
                
    def populate_lib_combo(self):
        for i in self.available_functions:
            name = self.fn_names[i]
            self.cmb_fit_function.addItem(name)
        
    def draw_data(self):
        if not self.data is None:
            x = self.data['time']
            y = self.data[self.trace_ids]
            self.canvas.draw_data(x, y)

    def fill_library(self):                
        for name in self.fn_dictionary:
            fn_ref = self.fn_dictionary[name][0]
            param_names = self.fn_dictionary[name][1]
            fn_str = self.fn_dictionary[name][2]
            self.library[name] = ff.ModellingFunction(name, fn_ref, param_names, fn_str)    
            
        