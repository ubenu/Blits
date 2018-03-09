'''
Created on 8 Mar 2018

@author: schilsm
'''
from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets

import copy as cp
import pandas as pd, numpy as np
from pandas.tools.plotting import df_ax

if __name__ == '__main__':
    pass


class SeriesLinkageDialog(widgets.QDialog):
    
    def __init__(self, parent, series_names, parameter_names):
        super(SeriesLinkageDialog, self).__init__(parent)
                
        self.setWindowTitle("Link parameters across series")
        main_layout = widgets.QHBoxLayout()
        self.setLayout(main_layout)
        
        self.button_box = widgets.QDialogButtonBox()
        self.tbl_links = widgets.QTableWidget()
        main_layout.addWidget(self.tbl_links)
        main_layout.addWidget(self.button_box)

        self.button_box.setOrientation(qt.Qt.Vertical)
        self.button_box.addButton(widgets.QDialogButtonBox.Cancel)
        self.button_box.addButton(widgets.QDialogButtonBox.Reset)
        self.button_box.addButton(widgets.QDialogButtonBox.Apply)
        self.button_box.addButton(widgets.QDialogButtonBox.Ok)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(widgets.QDialogButtonBox.Reset).clicked.connect(self.on_reset)
        
        self.series_names = series_names
        self.parameter_names = parameter_names
        self.tbl_index = ["Link all"]
        self.tbl_index.extend(series_names)
        print(self.tbl_index)
        self.set_table()
        
    def set_table(self):
        self.df_combos, self.df_groups = None, None
        nser, npar = len(self.series_names), len(self.parameter_names)
        self.df_combos = pd.DataFrame(index=self.series_names, columns=self.parameter_names)
        self.df_groups = pd.DataFrame(index=self.series_names, columns=self.parameter_names)
        self.sr_chk_series = pd.Series(index=self.parameter_names)
        self.tbl_links.setColumnCount(npar)
        self.tbl_links.setRowCount(nser+1)
        for col, pname in zip(range(npar), self.parameter_names):
            hhitem = widgets.QTableWidgetItem(pname)
            self.tbl_links.setHorizontalHeaderItem(col, hhitem)
            for row, sname in zip(range(nser+1), self.tbl_index):
                if row == 0:
                    wid = widgets.QWidget()
                    hlo = widgets.QVBoxLayout()
                    hlo.setAlignment(qt.Qt.AlignCenter)
                    cbox = widgets.QCheckBox()
                    cbox.setCheckState(qt.Qt.Unchecked)
                    cbox.setText("")
                    cbox.stateChanged.connect(self.on_cbox_changed)
                    self.sr_chk_series.loc[pname] = cbox
                    wid.setLayout(hlo)
                    hlo.addWidget(cbox)
                    self.tbl_links.setCellWidget(row, col, wid)
                else:                    
                    dbox = widgets.QComboBox()
                    dbox.addItems(self.series_names)
                    dbox.setEditable(False)
                    dbox.setCurrentText(sname)
                    dbox.currentIndexChanged.connect(self.on_table_item_changed)
                    self.df_combos.loc[sname, pname] = dbox
                    self.df_groups.loc[sname, pname] = {sname}
                    self.tbl_links.setCellWidget(row, col, dbox)
        self.tbl_links.setVerticalHeaderLabels(self.tbl_index) #(self.series_names)
        self.tbl_links.resizeColumnsToContents()
        self.tbl_links.resizeRowsToContents()
        
    def on_cbox_changed(self):
        param, sname = "", ""
        for pname, cbox in self.sr_chk_series.iteritems():
            if cbox is self.sender():
                param = pname
        sr_to_change = cp.deepcopy(self.df_combos.loc[:, param])
        for s_ind, sname in sr_to_change.iteritems():
            dbox = self.df_combos.loc[s_ind, param]
            dbox.currentIndexChanged.disconnect()
            if self.sender().checkState() == qt.Qt.Unchecked:
                dbox.setCurrentText(s_ind)  # PROTEST against using setCurrentText - wgy?
            else:
                dbox.setCurrentText(sname) 
            dbox.currentIndexChanged.connect(self.on_table_item_changed)
                
        
            
        
    def on_reset(self):
        self.set_table()
    
    def on_table_item_changed(self):
        self.df_groups = self.get_groups_from_table()
        for series, row in self.df_groups.iterrows(): # iterate over series
            for param, val, in row.iteritems(): # iterate over parameters
                dbox = self.df_combos.loc[series, param]
                dbox.currentIndexChanged.disconnect() # have to disconnect to avoid endless loop
                dbox.setCurrentText(val)
                dbox.currentIndexChanged.connect(self.on_table_item_changed)
                
    def get_groups_from_table(self):
        df_groups = pd.DataFrame(index=self.series_names, columns=self.df_combos.columns)#index=self.df_combos.index, columns=self.df_combos.columns)
        df_linkages = pd.DataFrame(index=df_groups.index, columns=df_groups.columns)
        for series, row in self.df_combos.iterrows(): # iterate over series
            for param, val, in row.iteritems(): # iterate over parameters
                df_linkages.loc[series, param] = val.currentText()
        
        for param, row in df_linkages.transpose().iterrows():
            x = row.index
            df_wf = pd.DataFrame(np.zeros((len(x), len(x))), index=x, columns=x, dtype=bool)
            for series, val in row.iteritems():
                df_wf.loc[series, series] = True # make the matrix reflexive
                if series != val:
                    df_wf.loc[series, val] = True
                    df_wf.loc[val, series] = True # make the matrix symmetrical
            # make matrix transitive (Warshall-Floyd)
            for k in range(len(x)):
                for i in range(len(x)):
                    for j in range(len(x)):
                        df_wf.iloc[i, j] = df_wf.iloc[i, j] or (df_wf.iloc[i, k] == 1 and df_wf.iloc[k, j] == 1)
            # Find the equivalence classes for this parameter 
            seen = []
            sr_equiv_clss = pd.Series(index=x)          
            for series0, row in df_wf.iterrows():
                for series1, val in row.iteritems():
                    if val:
                        if series1 not in seen:
                            sr_equiv_clss.loc[series1] = series0
                            seen.append(series1)
            df_groups.loc[:, param] = sr_equiv_clss
        return df_groups
                                        
"""
                         
    def get_linked_params_from_table(self):
"""
#         Returns an (n_series, n_params)-shaped array (with rows and columns parallel 
#         to self.series_names and self.fn_dictionary[fn][self.d_pnames], respectively)
#         of integers, in which linked parameters are grouped by their values.
#         Example for 4 curves and 3 parameters:
#               p0    p1    p2
#         c0    0     2     3
#         c1    0     2     4
#         c2    1     2     5
#         c3    1     2     6
#         indicates that parameter p0 is assumed to have the same value in 
#         curves c0 and c1, and in curves c2 and c3 (a different value), 
#         and that the value for p1 is the same in all curves, whereas
#         the value of p2 is different for all curves. 
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