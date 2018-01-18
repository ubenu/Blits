'''
Created on 18 Jan 2018

@author: schilsm
'''
from PyQt5 import QtCore as qt

class CruxTableModel(qt.QAbstractTableModel):
    
    def __init__(self, df_data):  
        super(CruxTableModel, self).__init__()
        self.df_data = df_data
        
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
            if role == qt.Qt.DisplayRole:
                return str(self.df_data.iloc[index.row(), index.column()])
            return qt.QVariant()
        return qt.QVariant()

    def setData(self, index, value, role):
        if index.isValid() and role == qt.Qt.EditRole:
            row, col = index.row(), index.column()
            if row in range(self.df_data.shape[0]) and col in range(self.df_data.shape[1]):
                print(type(self.df_data.iloc[row][col]))
                print(type(value))
                try:
                    self.df_data.iloc[row][col] = value
                    self.dataChanged.emit(index, index)
                    return True
                except ValueError:
                    return False
            return False
        return False
 
    def flags(self, index):
        flags = super(self.__class__,self).flags(index)
        flags |= qt.Qt.ItemIsEditable
        flags |= qt.Qt.ItemIsSelectable
        flags |= qt.Qt.ItemIsEnabled
        return flags             