'''
Created on 8 Mar 2018

@author: schilsm
'''
from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets

import pandas as pd, numpy as np
from pandas.tools.plotting import df_ax

if __name__ == '__main__':
    pass


class SeriesLinkageDialog(widgets.QDialog):
    
    def __init__(self, parent, series_names, parameter_names):
        super(SeriesLinkageDialog, self).__init__(parent)
        self.setModal(False)
                
        self.setWindowTitle("Link parameters across series")
        main_layout = widgets.QHBoxLayout()
        self.setLayout(main_layout)
        
        self.button_box = widgets.QDialogButtonBox()
        self.tbl_links = widgets.QTableWidget()
        main_layout.addWidget(self.tbl_links)
        main_layout.addWidget(self.button_box)

        self.button_box.setOrientation(qt.Qt.Vertical)
        self.button_box.addButton(widgets.QDialogButtonBox.Cancel)
        self.button_box.addButton(widgets.QDialogButtonBox.Ok)
        self.button_box.addButton(widgets.QDialogButtonBox.Apply)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        self.set_table(series_names, parameter_names)
        
        
        
    def set_table(self, series_names, parameter_names):
        nser, npar = len(series_names), len(parameter_names)
        self.df_combos = pd.DataFrame(index=series_names, columns=parameter_names)
        self.df_groups = pd.DataFrame(index=series_names, columns=parameter_names)
        self.tbl_links.setColumnCount(npar)
        self.tbl_links.setRowCount(nser)
        for col, pname in zip(range(npar), parameter_names):
            hhitem = widgets.QTableWidgetItem(pname)
            self.tbl_links.setHorizontalHeaderItem(col, hhitem)
            for row, sname in zip(range(nser), series_names):
                dbox = widgets.QComboBox()
                dbox.addItems(series_names)
                dbox.setEditable(False)
                dbox.setCurrentText(sname)
                dbox.currentIndexChanged.connect(self.on_table_item_changed)
                self.df_combos.loc[sname, pname] = dbox
                self.df_groups.loc[sname, pname] = {sname}
                self.tbl_links.setCellWidget(row, col, dbox)
        self.tbl_links.setVerticalHeaderLabels(series_names)
        
    def on_accept(self):
        pass
    
    def on_table_item_changed(self):
        box = self.sender()
        print(box.currentText())
        for sname in self.df_combos.index:
            for pname in self.df_combos.sname:
                if self.df_combos.loc[sname, pname] is box:
                    print(box.currentText())
            
#         for row, snames in zip(range(len(self.df_combos)), self.df_combos):
#             for col, param in zip(range(len(snames)), snames):
#                 if self.df_combos.iloc[row, col] is box:
#                     print(row, self.df_combos.iloc[row, col].currentText())
#                     for grp in self.df_groups.iloc[:, col]:
#                         print(grp)
                        
                         
    def get_linked_params_from_table(self):
        """
        Returns an (n_series, n_params)-shaped array (with rows and columns parallel 
        to self.series_names and self.fn_dictionary[fn][self.d_pnames], respectively)
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
        ncol_per_param = len(self.params_table_headers) - 2 #3
        if self.chk_global.checkState() == qt.Qt.Checked:
            ncol_per_param += 1                 

        funcname = self.cmb_fit_function.currentText()
        param_names = list(self.fn_dictionary[funcname][self.d_pnames])
        
        nparams = len(param_names)
        ncurves = len(self.series_names) 
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
                    if cname in self.series_names:
                        linked = self.tbl_params.cellWidget(irow, lloc).currentText()
                        cind = self.series_names.index(cname)
                        lind = self.series_names.index(linked)
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
                ind_params = np.empty_like(self.series_names, dtype=int)
                for i in eq_classes:
                    ind_params[i] = indpcount
                    indpcount += 1
                links[pcount] = ind_params
                pcount += 1
        
        selected = [self.series_names.index(name) for name in series_names]        
        return links.transpose()[selected]


"""


import functions.function_defs as fdefs


class FunctionSelectionDialog(widgets.QDialog):
    
    def __init__(self, parent, n_axes=np.inf, selected_fn_name=""):
        super(FunctionSelectionDialog, self).__init__(parent)
        self.setModal(False)
                
        self.setWindowTitle("Select modelling function")
        main_layout = widgets.QVBoxLayout()
        self.button_box = widgets.QDialogButtonBox()
        self.button_box.addButton(widgets.QDialogButtonBox.Cancel)
        self.button_box.addButton(widgets.QDialogButtonBox.Ok)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        table_label = widgets.QLabel("Available functions")
        self.tableview = widgets.QTableView()
        table_label.setBuddy(self.tableview)
        self.model = FunctionLibraryTableModel(n_axes=n_axes)
        self.tableview.setModel(self.model)
        self.tableview.setSelectionBehavior(widgets.QAbstractItemView.SelectRows)
        self.tableview.setSelectionMode(widgets.QAbstractItemView.SingleSelection)
        self.tableview.doubleClicked.connect(self.accept)
        
        self.selected_fn_name = selected_fn_name
        if self.selected_fn_name != "":
            self.tableview.selectRow(self.findItem(self.selected_fn_name).row())
                
        main_layout.addWidget(self.tableview)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)
        
"""        