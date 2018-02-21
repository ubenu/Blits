'''
Created on 18 Jan 2018

@author: schilsm
'''
from PyQt5 import QtCore as qt
import copy as cp
from bokeh.layouts import column

class CruxTableModel(qt.QAbstractTableModel):
    
    def __init__(self, df_data, checkable_cols=[]):  
        super(CruxTableModel, self).__init__()
        self.df_data = df_data
        self.df_checks = cp.deepcopy(self.df_data)
        self.df_checks.iloc[:,:] = False
        self.checkable_columns = checkable_cols
        
    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        if role == qt.Qt.DisplayRole:
            if orientation == qt.Qt.Horizontal:
                return self.df_data.columns[section]
            elif orientation == qt.Qt.Vertical:
                return self.df_data.index[section]
            return qt.QVariant()
        return qt.QVariant()

    def rowCount(self, index=qt.QModelIndex()):
        return self.df_data.shape[0]

    def columnCount(self, index=qt.QModelIndex()):
        return self.df_data.shape[1]
    
    def data(self, index, role=qt.Qt.DisplayRole):
        if index.isValid():
            if role in (qt.Qt.DisplayRole, qt.Qt.EditRole):
                return str(self.df_data.iloc[index.row(), index.column()])
            if not len(self.checkable_columns) < 1:
                if role == qt.Qt.CheckStateRole and index.column() in self.checkable_columns:
                    if self.df_checks.iloc[index.row(), index.column()]:
                        return qt.Qt.Checked
                    else:
                        return qt.Qt.Unchecked
            return qt.QVariant()
        return qt.QVariant()

    def setData(self, index, value, role):
        # Setting data has to be done via .loc to avoid working on a copy; 
        # see: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
        if index.isValid():
            row, col = self.df_data.index[index.row()], self.df_data.columns[index.column()]
            if role == qt.Qt.EditRole:
                try:
                    if isinstance(self.df_data.loc[row, col], str) and self.df_data.loc[row, col] != "":
                        self.df_data.loc[row, col] = value 
                        self.dataChanged.emit(index, index)
                        return True
                    elif isinstance(self.df_data.loc[row, col], float):
                        self.df_data.loc[row, col] = float(value) 
                        self.dataChanged.emit(index, index)
                        return True                    
                    return False
                except Exception as e:
                    print(e)
                    return False
            if role == qt.Qt.CheckStateRole:
                if self.data(index, qt.Qt.CheckStateRole) == qt.Qt.Checked:
                    self.df_checks.loc[row, col] = False
                else:
                    self.df_checks.loc[row, col] = True
                self.dataChanged.emit(index, index)
                return True
            return False
        return False
    
    def replace_all_data(self, df_data):
        for row in range(self.rowCount()):
            for col in range(self.columnCount()):
                self.setData((row,col), df_data.iloc[row, col])
                
 
    def flags(self, index):
        flags = super(self.__class__,self).flags(index)
        flags |= qt.Qt.ItemIsEditable
        flags |= qt.Qt.ItemIsSelectable
        flags |= qt.Qt.ItemIsEnabled
        flags |= qt.Qt.ItemIsUserCheckable
        return flags             