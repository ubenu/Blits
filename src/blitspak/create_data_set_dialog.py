'''
Created on 15 Jan 2018

@author: schilsm
'''
from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets

import pandas as pd, numpy as np

class CreateDataSetDialog(widgets.QDialog):
    '''
    classdocs
    '''

    def __init__(self, parent, selected_fn=None):
        '''
        Constructor
        '''
        super(CreateDataSetDialog, self).__init__(parent)
        self.setWindowTitle("Create a data set")
        main_layout = widgets.QVBoxLayout()
        self.button_box = widgets.QDialogButtonBox()
        self.button_box.addButton(widgets.QDialogButtonBox.Cancel)
        self.button_box.addButton(widgets.QDialogButtonBox.Ok)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        main_layout.addWidget(self.button_box)
        
        self.selected_fn = selected_fn
        if self.selected_fn is None:
            print("Please select a modelling function first")
            
        print('Function selected')
        
        
            
        
    def accept(self):
        widgets.QDialog.accept(self)
        
    def reject(self):
        widgets.QDialog.reject(self)
        
                        