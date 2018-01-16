"""
Blits:
Created on 23 May 2017
Original Blivion:
Created on Tue Oct 25 13:11:32 2016

@author: Maria Schilstra
"""

#from PyQt5.uic import loadUiType

from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets
from PyQt5 import QtGui as gui

import pandas as pd, numpy as np


from matplotlib.widgets import SpanSelector
from blitspak.blits_mpl import MplCanvas, NavigationToolbar
from blitspak.blits_data import BlitsData
from blitspak.scrutinize_dialog import ScrutinizeDialog
from blitspak.function_dialog import FunctionSelectionDialog
from blitspak.create_data_set_dialog import CreateDataSetDialog
#import blitspak.blits_ui as ui
from PyQt5.uic import loadUiType
Ui_MainWindow, QMainWindow = loadUiType('..\\..\\Resources\\UI\\blits.ui')

# Original:
# To avoid using .ui file (from QtDesigner) and loadUIType, 
# created a python-version of the .ui file using pyuic5 from command line
# Here: pyuic5 blits.ui -o blits_ui.py
# Also: cannot (easily) use .qrc file, so need to create _rc.py file
# with icon definitions: pyrcc5 -o blits_rc.py blits.qrc
# Then import .py package, as below.
# (QMainWindow is a QtWidget; UI_MainWindow is generated by the converted .ui)



class Main(QMainWindow, Ui_MainWindow):
    
    NSTATES = 6
    START, DATA_ONLY, FUNCTION_ONLY, READY_FOR_FITTING, FITTED, REJECT = range(NSTATES)
#     readiness = (1, 2)
#     DATA_SET, FUNCTION_SET = readiness

    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)

        self.scrutinize_dialog = None
        self.function_dialog = None
        self.create_data_set_dialog = None
        
        self.canvas = MplCanvas(self.mpl_window)
        self.plot_toolbar = NavigationToolbar(self.canvas, self.mpl_window)
        self.mpl_layout.addWidget(self.canvas)
        self.frm_show_axis = widgets.QFrame()
        self.axis_layout = widgets.QHBoxLayout()
        self.frm_show_axis.setLayout(self.axis_layout)
        self.mpl_layout.addWidget(self.frm_show_axis)
        self.mpl_layout.addWidget(self.plot_toolbar)

        self.action_open.triggered.connect(self.on_open)
        self.action_create.triggered.connect(self.on_create)
        self.action_close.triggered.connect(self.on_close_data)
        self.action_save.triggered.connect(self.on_save)
        self.action_select_function.triggered.connect(self.on_select_function)
        self.action_analyze.triggered.connect(self.on_analyze)
        self.action_quit.triggered.connect(self.close)     

        self.span = SpanSelector(self.canvas.data_plot, self.on_select_span, 
        'horizontal', useblit=True, rectprops=dict(alpha=0.5, facecolor='red'))
        
        self.blits_data = BlitsData()
        self.file_name = ""
        self.file_path = ""
        self.phase_number = 0
        self.phase_name = "Phase"
        self.phase_list = []
        self.current_state = self.START
        self.current_function = None
        
        self.update_ui()
        
    def on_open(self):
        if self.current_state in (self.START, self.FUNCTION_ONLY, ):
            file_path = widgets.QFileDialog.getOpenFileName(self, 
            "Open Data File", "", "CSV data files (*.csv);;All files (*.*)")[0]
            if file_path:
                self.file_path = file_path
                info = qt.QFileInfo(file_path)
                self.blits_data.file_name = info.fileName()
                self.blits_data.import_data(file_path)
                self.canvas.set_colours(self.blits_data.series_names.tolist())
                for key in self.blits_data.series_names:
                    series = self.blits_data.series_dict[key]
                    x = series.iloc[:, 0]
                    y = series['y']
                    self.canvas.draw_series(key, x, y)
                if self.current_state == self.START:
                    self.current_state = self.DATA_ONLY
                elif self.current_state == self.FUNCTION_ONLY:
                    self.current_state = self.READY_FOR_FITTING
                self.update_ui()
            
    def on_create(self):  
        if self.current_state in (self.FUNCTION_ONLY, ):
            self.create_data_set_dialog = CreateDataSetDialog(None, self.current_function)
            if self.create_data_set_dialog.exec() == widgets.QDialog.Accepted:
                print('Simulated data created')
                self.current_state = self.READY_FOR_FITTING
                self.update_ui()
            else:
                print('No data created')

    def on_close_data(self):
        if self.current_state in (self.DATA_ONLY, self.READY_FOR_FITTING, self.FITTED, ):
            self.blits_data = BlitsData()
            self.canvas.clear_figure()
            if self.current_state == self.DATA_ONLY:
                self.current_state = self.START
            else:
                self.current_state = self.FUNCTION_ONLY
            self.update_ui()

    def on_select_function(self):
        if self.current_state in range(self.NSTATES):  # should work from all states
            name = ""
            if not self.current_state in (self.START, self.DATA_ONLY):  # a current function exists
                name = self.current_function.name
            self.function_dialog = FunctionSelectionDialog(self, name)
            if self.function_dialog.exec() == widgets.QDialog.Accepted:
                self.current_function = self.function_dialog.get_selected_function()
                self.model = ParametersTableModel(self.current_function)
                self.lbl_fn_name.setText("Selected function: " + self.current_function.name)
                self.txt_description.setText(self.current_function.long_description)
                self.table_view.setModel(self.model)
                if self.current_state in (self.START, self.FUNCTION_ONLY):
                    self.current_state = self.FUNCTION_ONLY
                else:
                    self.current_state = self.READY_FOR_FITTING
                self.update_ui()
            
    def on_analyze(self):
        if self.current_state in (self.READY_FOR_FITTING, ):
#         if self.action_analyze.isChecked():
#             self.plot_toolbar.switch_off_pan_zoom()
#             self.span.set_active(True)   
#         else:
#             self.span.set_active(False)  
            self.current_state = self.FITTED
            self.update_ui()

    def on_save(self):
        file_path = ""
        if self.current_state in (self.DATA_ONLY, self.READY_FOR_FITTING, ):
            file_path = widgets.QFileDialog.getSaveFileName(self, 
            "Save data", "", "CSV data files (*.csv);;All files (*.*)")[0]
        if self.current_state in (self.FITTED, ):
            file_path = widgets.QFileDialog.getSaveFileName(self, 
            "Save data and fit", "", "CSV data files (*.csv);;All files (*.*)")[0]
        if file_path:
            pass
#             self.blits_data.export_results(file_path)
            
    def on_select_span(self, xmin, xmax):
        self.span.set_active(False)
        if (xmin != xmax):
            phase_id = self.phase_name + '{:0d}'.format(self.phase_number + 1)
            self.scrutinize_dialog = ScrutinizeDialog(main, xmin, xmax) 
            flags = self.scrutinize_dialog.windowFlags() | qt.Qt.WindowMinMaxButtonsHint
            self.scrutinize_dialog.setWindowFlags(flags)
            if self.scrutinize_dialog.show() == widgets.QDialog.Accepted:
                self.phase_number += 1
                self.phase_list.append(phase_id)
                model = self.scrutinize_dialog.current_function
                model_expr = self.scrutinize_dialog.fn_dictionary[model][self.scrutinize_dialog.d_expr]
                m_string = model + ': ' + model_expr
                tbl_results = self.scrutinize_dialog.tbl_results
                self._create_results_tab(phase_id, m_string, tbl_results)
            self.action_analyze.setChecked(False)
            self.on_analyze()
                
    def _create_results_tab(self, phase_id, model_string, results_table):
        new_tab = widgets.QWidget()
        lo = widgets.QVBoxLayout(new_tab)
        x = widgets.QLabel(model_string)
        lo.addWidget(x)
        lo.addWidget(results_table)
        new_tab.setLayout(lo)
        
        self.tabWidget.addTab(new_tab, phase_id)
        
    def line_icon(self, color):
        pixmap = gui.QPixmap(50,10)
        pixmap.fill(gui.QColor(color))
        icon = gui.QIcon(pixmap)
        return icon  
    
    def circle_icon(self, color):
        pix = gui.QPixmap(30,30)
        pix.fill(gui.QColor("transparent"))
        paint = gui.QPainter()
        paint.begin(pix)
        paint.setBrush(gui.QColor(color))
        paint.setPen(gui.QColor("transparent"))
        paint.drawEllipse(0,0,30,30)
        paint.end()
        icon = gui.QIcon(pix)
        return icon        

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False  
        
    def update_ui(self):
        if self.current_state == self.START:
            self.action_open.setEnabled(True)
            self.action_create.setEnabled(False)
            self.action_close.setEnabled(False)
            self.action_save.setEnabled(False)
            self.action_select_function.setEnabled(True)
            self.action_analyze.setEnabled(False)
            self.action_quit.setEnabled(True)     
            self.span.set_active(False)
        elif self.current_state == self.DATA_ONLY:
            self.action_open.setEnabled(False)
            self.action_create.setEnabled(False)
            self.action_close.setEnabled(True)
            self.action_save.setEnabled(True)
            self.action_select_function.setEnabled(True)
            self.action_analyze.setEnabled(False)
            self.action_quit.setEnabled(True)     
            self.span.set_active(False)
        elif self.current_state == self.FUNCTION_ONLY:
            self.action_open.setEnabled(True)
            self.action_create.setEnabled(True)
            self.action_close.setEnabled(False)
            self.action_save.setEnabled(False)
            self.action_select_function.setEnabled(True)
            self.action_analyze.setEnabled(False)
            self.action_quit.setEnabled(True)     
            self.span.set_active(False)
        elif self.current_state == self.READY_FOR_FITTING:
            self.action_open.setEnabled(False)
            self.action_create.setEnabled(False)
            self.action_close.setEnabled(True)
            self.action_save.setEnabled(True)
            self.action_select_function.setEnabled(True)
            self.action_analyze.setEnabled(True)
            self.action_quit.setEnabled(True)     
            self.span.set_active(False)
        elif self.current_state == self.FITTED:
            self.action_open.setEnabled(False)
            self.action_create.setEnabled(False)
            self.action_close.setEnabled(True)
            self.action_save.setEnabled(True)
            self.action_select_function.setEnabled(True)
            self.action_analyze.setEnabled(True)
            self.action_quit.setEnabled(True)     
            self.span.set_active(False)
        else:
            print('Illegal state')
                                          
        
class ParametersTableModel(qt.QAbstractTableModel):
    
    def __init__(self, modfunc):  
        super(ParametersTableModel, self).__init__() 
        self.modfunc = modfunc
        x1 = np.linspace(0, 10.0, 0.1)
        y1 = np.sin(x1)
        y2 = np.cos(2*x1)
        d = {'X1': x1, 'Y1': y1, 'X2': x1, 'Y2': y2}
        self.dummydata = pd.DataFrame(d)
        
    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        # Implementation of super.headerData
        if role == qt.Qt.TextAlignmentRole:
            if orientation == qt.Qt.Horizontal:
                return qt.QVariant(int(qt.Qt.AlignLeft|qt.Qt.AlignVCenter))
            return qt.QVariant(int(qt.Qt.AlignRight|qt.Qt.AlignVCenter))
        if role != qt.Qt.DisplayRole:
            return qt.QVariant()
        if orientation == qt.Qt.Vertical:
            for i in range(len(self.modfunc.parameters)):
                if section == i:
                    return qt.QVariant(self.modfunc.parameters[i])
        if orientation == qt.Qt.Horizontal:
            pass
        return qt.QVariant(int(section + 1))

    def rowCount(self, index=qt.QModelIndex()):
        return len(self.modfunc.parameters) 

    def columnCount(self, index=qt.QModelIndex()):
        return int(len(self.dummydata.columns)/2)
    
    def data(self, index, role=qt.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self.modfunc.parameters)):
            return qt.QVariant()
        column, row = index.column(), index.row()
        if role == qt.Qt.DisplayRole:
            return qt.QVariant(str(row) + ", " + str(column) )
        return qt.QVariant()        
  

# Standard main loop code
if __name__ == '__main__':
    import sys
#    sys.tracbacklimit = 10
    app = widgets.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())


### Old
#     def _write_results(self):
#         r = self.blits_data.results
#         tbr = self.tblResults
#         tbr.setColumnCount(len(r.columns))
#         tbr.setRowCount(len(r.index))
#         tbr.setVerticalHeaderLabels(r.index)
#         tbr.setHorizontalHeaderLabels(r.columns)
#         for i in range(len(r.index)):
#             for j in range(len(r.columns)):
#                 tbr.setItem(i,j,widgets.QTableWidgetItem(str(r.iat[i, j])))
# 
#         if self.blits_data.results_acquired['fractional saturation']:
#             p = self.blits_data.get_fractional_saturation_params_dataframe()
#             tbp = self.tblFitParams
#             tbp.setColumnCount(len(p.columns))
#             tbp.setRowCount(len(p.index))
#             tbp.setVerticalHeaderLabels(p.index)
#             tbp.setHorizontalHeaderLabels(p.columns)
#             for i in range(len(p.index)):
#                 for j in range(len(p.columns)):
#                     tbp.setItem(i,j,widgets.QTableWidgetItem(str(p.iat[i, j])))
#             
#             f = self.blits_data.get_fractional_saturation_curve()
#             tbf = self.tblFittedCurve
#             tbf.setColumnCount(len(f.columns))
#             tbf.setRowCount(len(f.index))
#             tbf.setHorizontalHeaderLabels(f.columns)
#             for i in range(len(f.index)):
#                 for j in range(len(f.columns)):
#                     tbf.setItem(i,j,widgets.QTableWidgetItem(str(f.iat[i, j])))        


