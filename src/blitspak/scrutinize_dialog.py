'''
Created on 6 Jun 2017

@author: SchilsM
'''
#from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets
from blitspak.blits_mpl import MplCanvas, NavigationToolbar

from functions.framework import FunctionsFramework as ff
from functions.framework import ModellingFunction as mf
import functions.function_defs as fd

#import scrutinize_dialog_ui as ui

from PyQt5.uic import loadUiType
Ui_ScrutinizeDialog, QDialog = loadUiType('..\\..\\Resources\\UI\\scrutinize_dialog.ui')

class ScrutinizeDialog(widgets.QDialog, Ui_ScrutinizeDialog):

    fn_dictionary = {"Average": (fd.fn_average, 
                                 ('a',), 
                                 "average(trace)"),
                     "Straight line": (fd.fn_straight_line, 
                                       ('a', 'b'), 
                                       "a+b*x"), 
                     "Single exponential": (fd.fn_1exp, 
                                            ('a0', 'a1', 'k1'), 
                                            "a0 + a1*np.exp(-x*k1)"),
                     "Double exponential": (fd.fn_2exp, 
                                            ('a0', 'a1', 'k1', 'a2', 'k2'), 
                                            "a0 + a1*np.exp(-x*k1) + a2*np.exp(-x*k2)"), 
                     "Triple exponential": (fd.fn_3exp, 
                                            ('a0', 'a1', 'k1', 'a2', 'k2', 'a3', 'k3'), 
                                            "a0 + a1*np.exp(-x*k1) + a2*np.exp(-x*k2) + a3*np.exp(-x*k3)"),
                     "Michaelis-Menten equation": (fd.fn_mich_ment, 
                                                   ('km', 'vmax'), 
                                                   "vmax * x / (km + x)"),
                     "Competitive inhibition equation": (fd.fn_comp_inhibition, 
                                                         ('km', 'ki', 'vmax'), 
                                                         "vmax * x[0] / (km * (1.0 + x[1] / ki) + x[0])"), 
                     "Hill equation": (fd.fn_hill, 
                                       ('ymax', 'xhalf', 'h'), 
                                       "ymax / ((xhalf/x)^h + 1.0)"),
                     }

    def __init__(self, parent, trace_segments):
        '''
        Constructor
        '''
        super(widgets.QDialog, self).__init__(parent)
        self.setupUi(self)
        
        self.library = {}
        self.fill_library()
        self.populate_lib_combo()
        self.cmb_fit_function.setCurrentIndex(0)
        self.on_current_index_changed(self.cmb_fit_function.currentIndex())
        
        self.cmb_fit_function.currentIndexChanged.connect(self.on_current_index_changed)

        self.canvas = MplCanvas(self.mpl_window)
        self.mpl_layout.addWidget(self.canvas)
        self.plot_toolbar = NavigationToolbar(self.canvas, self.mpl_window)
        self.mpl_layout.addWidget(self.plot_toolbar)
        
        self.data = trace_segments
        self.draw_data()
        
    def on_current_index_changed(self, index):
        self.txt_function.clear()
        name = self.cmb_fit_function.currentText() 
        self.txt_function.setText(self.library[name].fn_str)
        
    def populate_lib_combo(self):
        for name in self.library:
            self.cmb_fit_function.addItem(name)
        
    def draw_data(self):
        if not self.data is None:
            x = self.data['time']
            y = self.data[self.data.columns[self.data.columns != 'time']]
            self.canvas.draw_data(x, y)

    def fill_library(self):                
        for name in self.fn_dictionary:
            fn_ref = self.fn_dictionary[name][0]
            param_names = self.fn_dictionary[name][1]
            fn_str = self.fn_dictionary[name][2]
            self.library[name] = mf(name, fn_ref, param_names, fn_str)    
            
        