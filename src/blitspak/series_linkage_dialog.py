'''
Created on 8 Mar 2018

@author: schilsm
'''
from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets

import copy as cp, pandas as pd, numpy as np

if __name__ == '__main__':
    pass


class SeriesLinkageDialog(widgets.QDialog):
    
    NSTATES = 2
    NO_DATA, HAS_DATA = range(NSTATES)
    
    def __init__(self, parent, linkage_matrix):
        super(SeriesLinkageDialog, self).__init__(parent)
        self.setAttribute(qt.Qt.WA_DeleteOnClose)
                
        self.setWindowTitle("Link parameters across series")
        main_layout = widgets.QHBoxLayout()
        self.setLayout(main_layout)
        
        self.button_box = widgets.QDialogButtonBox()
        self.tbl_links = widgets.QTableWidget()
        main_layout.addWidget(self.tbl_links)
        main_layout.addWidget(self.button_box)

        self.button_box.setOrientation(qt.Qt.Vertical)
        self.button_box.addButton(widgets.QDialogButtonBox.Reset)
        self.button_box.button(widgets.QDialogButtonBox.Reset).clicked.connect(self.on_reset)
        
        self.state = self.NO_DATA
        self.set_ui(linkage_matrix)
            
    def set_ui(self, linkage_matrix):
        if linkage_matrix is not None:
            self.state = self.HAS_DATA
            self.series_names = linkage_matrix.index
            self.parameter_names = linkage_matrix.columns
            self.df_groups = cp.deepcopy(linkage_matrix)
            self.create_containers()
            self.tbl_index = np.concatenate((["Link all"], self.series_names)).tolist()
            self.populate_table()
        else:
            self.series_names = None
            self.parameter_names = None
            self.df_groups = None
            self.df_combos = None
            self.sr_checks = None
            self.tbl_index = None
            
                        
    def on_chk_all(self):
        if self.state == self.HAS_DATA:
            pname = ""
            for p, cbox in self.sr_checks.iteritems():
                if cbox is self.sender():
                    pname = p
            for sname in self.series_names:
                if self.sender().checkState() == qt.Qt.Checked: # set all the name of the first series
                    self.df_groups.loc[sname, pname] = self.series_names[0]
                else:
                    self.df_groups.loc[sname, pname] = sname # set all to the original names
            self.set_table()
                
    def on_reset(self):
        """
        Sets the combo-boxes to the original values (via df_groups)
        and un-checks the parameter check-boxes
        """
        if self.state == self.HAS_DATA:
            for pname in self.parameter_names:
                cbox = self.sr_checks.loc[pname]
                cbox.disconnect()
                cbox.setCheckState(qt.Qt.Unchecked)
                cbox.stateChanged.connect(self.on_chk_all)
            for sname in self.series_names:
                for pname in self.parameter_names:
                    self.df_groups.loc[sname, pname] = sname
            self.set_table()
    
    def on_table_item_changed(self):
        if self.state == self.HAS_DATA:
            param = ""
            for sname, row in self.df_combos.iterrows():
                for pname, box, in row.iteritems():
                    if self.sender() is box:
                        param = pname
                        self.df_groups.loc[sname, pname] = box.currentText()
            self.rationalise_groups(param)
            self.set_table()
        
    def create_containers(self):
        """
        Creates df_combos (pandas DataFrame) and sr_checks (pandas Series)
        and their content (combo-boxes and unchecked check-boxes, respectively)
        """
        if self.state == self.HAS_DATA:
            self.df_combos = pd.DataFrame(index=self.series_names, columns=self.parameter_names)
            self.sr_checks = pd.Series(index=self.parameter_names)
            for sname in self.series_names:
                for pname in self.parameter_names:
                    linked_name = self.df_groups.loc[sname, pname]
                    cbox = widgets.QCheckBox()
                    cbox.setCheckState(qt.Qt.Unchecked)
                    cbox.setText("")
                    cbox.stateChanged.connect(self.on_chk_all)
                    self.sr_checks.loc[pname] = cbox
                    dbox = widgets.QComboBox()
                    dbox.addItems(self.series_names)
                    dbox.setEditable(False)
                    dbox.setCurrentText(linked_name)
                    dbox.currentIndexChanged.connect(self.on_table_item_changed)
                    self.df_combos.loc[sname, pname] = dbox
        
    def populate_table(self):
        """
        Creates the linkage table from the data in the containers
        """
        if self.state == self.HAS_DATA:
            nrows, npar = len(self.tbl_index), len(self.parameter_names)
            self.tbl_links.setRowCount(nrows)
            self.tbl_links.setColumnCount(npar)
            for col, pname in zip(range(npar), self.parameter_names):
                hhitem = widgets.QTableWidgetItem(pname)
                self.tbl_links.setHorizontalHeaderItem(col, hhitem)
                for row, sname in zip(range(nrows), self.tbl_index):
                    if row == 0:
                        # create a widget and layout to centre the check-box
                        wid = widgets.QWidget()
                        hlo = widgets.QVBoxLayout()
                        hlo.setAlignment(qt.Qt.AlignCenter)
                        wid.setLayout(hlo)
                        cbox = self.sr_checks.loc[pname]
                        hlo.addWidget(cbox)
                        self.tbl_links.setCellWidget(row, col, wid)
                    else:   
                        dbox = self.df_combos.loc[sname, pname]
                        self.tbl_links.setCellWidget(row, col, dbox)
            self.tbl_links.setVerticalHeaderLabels(self.tbl_index)
            self.tbl_links.resizeColumnsToContents()
            self.tbl_links.resizeRowsToContents()
        else:
            self.tbl_links.setRowCount(0)
            self.tbl_links.setColumnCount(0)
        
    def set_table(self):
        """
        Sets combo-boxes to the current values in df_groups        
        """
        if self.state == self.HAS_DATA:
            for sname in self.series_names:
                for pname in self.parameter_names:
                    sgrp = self.df_groups.loc[sname, pname]
                    dbox = self.df_combos.loc[sname, pname]
                    if dbox.currentText() != sgrp:
                        dbox.currentIndexChanged.disconnect()
                        dbox.setCurrentText(sgrp)
                        dbox.currentIndexChanged.connect(self.on_table_item_changed)
        
    def rationalise_groups(self, param):
        if self.state == self.HAS_DATA and param != '':
            col = self.df_groups.loc[:, param]
            x = col.index
            df_wf = pd.DataFrame(np.zeros((len(x), len(x))), index=x, columns=x, dtype=bool) # set up the matrix
            for series, val in col.iteritems():
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
            self.df_groups.loc[:, param] = sr_equiv_clss
                
    def get_unique_params_matrix(self):
        """
        Returns an (n_series, n_params)-shaped array (with rows and columns parallel 
        to self.series_names and self.fn_dictionary[fn][self.d_pnames], respectively)
        of integers, in which linked parameters are grouped by their values.
        Example for 4 curves and 3 parameters:
              p0        p1        p2
        c0    c0_p0     c0_p1     c0_p2
        c1    c0_p0     c0_p1     c1_p2
        c2    c2_p0     c0_p1     c2_p2
        c3    c3_p0     c0_p1     c3_p2
        indicates that parameter p0 is assumed to have the same value in 
        curves c0 and c1, and in curves c2 and c3 (a different value), 
        and that the value for p1 is the same in all curves, whereas
        the value of p2 is different for all curves. 
        """
        if self.state == self.HAS_DATA:
            unique_params = pd.DataFrame(index=self.df_combos.index, columns=self.df_combos.columns)
            for s_ind, s in self.df_groups.iterrows():
                for p_ind, grp in s.iteritems():
                    unique_params.loc[s_ind, p_ind] = grp + '_' + p_ind
            print(unique_params)
            return unique_params
        return None
                                        
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


"""