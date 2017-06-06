'''
Created on 6 Jun 2017

@author: SchilsM
'''
#from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets

#import scrutinize_dialog_ui as ui

from PyQt5.uic import loadUiType
Ui_ScrutinizeDialog, QDialog = loadUiType('scrutinize_dialog.ui')

class ScrutinizeDialog(widgets.QDialog, Ui_ScrutinizeDialog):


    def __init__(self, parent):
        '''
        Constructor
        '''
        super(widgets.QDialog, self).__init__(parent)
        self.setupUi(self)
        