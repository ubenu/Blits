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

from matplotlib.widgets import SpanSelector
from blitspak.blits_mpl import MplCanvas, NavigationToolbar
from blitspak.blits_data import BlitsData
from blitspak.scrutinize_dialog import ScrutinizeDialog
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
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)

        self.scrutinize_dialog = None
        
        self.canvas = MplCanvas(self.mpl_window)
        self.plot_toolbar = NavigationToolbar(self.canvas, self.mpl_window)
        self.mpl_layout.addWidget(self.canvas)
        self.mpl_layout.addWidget(self.plot_toolbar)

        self.span = SpanSelector(self.canvas.data_plot, self.on_select_span, 
        'horizontal', useblit=True, rectprops=dict(alpha=0.5, facecolor='red'))
        
        self.action_open.triggered.connect(self.on_open)
        self.action_save.triggered.connect(self.on_save)
        self.action_close.triggered.connect(self.on_close)
        self.action_quit.triggered.connect(self.close)     
        self.action_scrutinize.triggered.connect(self.on_scrutinize)

        self.blits_data = BlitsData()
        self.file_name = ""
        self.file_path = ""
        self.phase_number = 0
        self.phase_name = "Phase"
        self.phase_list = []

        self.span.set_active(False)

        self._data_open = False
        self._scrutinizing = False
        
        self.action_open.setEnabled(True)
        self.action_save.setEnabled(False)
        self.action_close.setEnabled(False)
        self.action_quit.setEnabled(True)
        self.action_scrutinize.setEnabled(False)

                
    def on_open(self):
        file_path = widgets.QFileDialog.getOpenFileName(self, 
        "Open Data File", "", "CSV data files (*.csv);;All files (*.*)")[0]
        if file_path:
            self.file_path = file_path
            info = qt.QFileInfo(file_path)
            self.blits_data.file_name = info.fileName()
            if self._data_open:
                self.on_close()
            self.blits_data.import_data(file_path)
            x = self.blits_data.get_data_x()
            y = self.blits_data.get_data_y()
            self.action_scrutinize.setEnabled(True)
            self.canvas.draw_data(x, y)
            self._data_open = True
            self._scrutinizing = False
            
            self.action_open.setEnabled(True)
            self.action_save.setEnabled(False)
            self.action_close.setEnabled(True)
            self.action_quit.setEnabled(True)
            self.action_scrutinize.setEnabled(True)
            
    def on_save(self):
        file_path = widgets.QFileDialog.getSaveFileName(self, 
        "Save Results File", "", "CSV data files (*.csv);;All files (*.*)")[0]
        if file_path:
            self.blits_data.export_results(file_path)
        
    def on_close(self):
        self.blits_data = BlitsData()
        self.canvas.clear_figure()
        self._data_open = False
        self._scrutinizing = False
        
        self.action_open.setEnabled(True)
        self.action_save.setEnabled(False)
        self.action_close.setEnabled(False)
        self.action_quit.setEnabled(True)
        self.action_scrutinize.setEnabled(False)
        
    def on_scrutinize(self):
        if self.action_scrutinize.isChecked():
            self.plot_toolbar.switch_off_pan_zoom()
            self._scrutinizing = True
            self.span.set_active(True)   
        else:
            self._scrutinizing = False
            self.span.set_active(False)             
        
    def on_select_span(self, xmin, xmax):
        self.span.set_active(False)
        if self._scrutinizing:  
            if (xmin != xmax):
                phase_id = self.phase_name + '{:0d}'.format(self.phase_number + 1)
                self.scrutinize_dialog = ScrutinizeDialog(main, xmin, xmax) 
                flags = self.scrutinize_dialog.windowFlags() | qt.Qt.WindowMinMaxButtonsHint
                self.scrutinize_dialog.setWindowFlags(flags)
                if self.scrutinize_dialog.show() == widgets.QDialog.Accepted:
                    print(self.scrutinize_dialog.data)
                    self.phase_number += 1
                    self.phase_list.append(phase_id)
                    model = self.scrutinize_dialog.current_function
                    model_expr = self.scrutinize_dialog.fn_dictionary[model][self.scrutinize_dialog.d_expr]
                    m_string = model + ': ' + model_expr
                    tbl_results = self.scrutinize_dialog.tbl_results
                    self._create_results_tab(phase_id, m_string, tbl_results)
                self.action_scrutinize.setChecked(False)
                self.on_scrutinize()
                
    def _create_results_tab(self, phase_id, model_string, results_table):
        new_tab = widgets.QWidget()
        lo = widgets.QVBoxLayout(new_tab)
        x = widgets.QLabel(model_string)
        lo.addWidget(x)
        lo.addWidget(results_table)
        new_tab.setLayout(lo)
        
        self.tabWidget.addTab(new_tab, phase_id)
        
            
    def _draw_results(self):
        x = self.blits_data.get_data_x()
        y = self.blits_data.get_data_y()
        self.canvas.draw_data(x, y)
                                
    def _draw_analysis(self):
        if self.blits_data.results_acquired['baseline']:
            if self.blits_data.results_acquired['loaded']:
                load = self.blits_data.results['Sugar loading']
                self.canvas.draw_sugar_loading(load)
                if self.blits_data.results_acquired['association']:
                    self.blits_data.set_fractional_saturation_results()
                    if self.blits_data.results_acquired['fractional saturation']:
                        params = self.blits_data.fractional_saturation_params
                        mask = self.blits_data.results['success'] == 1.0
                        obs = self.blits_data.results[['Sugar loading', 'Amplitude (obs)',
                                                         'Amplitude (calc)']][mask]
                        fit = self.blits_data.get_fractional_saturation_curve()
                        res = obs['Amplitude (obs)'] - obs['Amplitude (calc)']
                        self.canvas.draw_fractional_saturation(obs['Sugar loading'], obs['Amplitude (obs)'],
                                                               res, fit['x'], fit['y'], params)

    def _write_results(self):
        r = self.blits_data.results
        tbr = self.tblResults
        tbr.setColumnCount(len(r.columns))
        tbr.setRowCount(len(r.index))
        tbr.setVerticalHeaderLabels(r.index)
        tbr.setHorizontalHeaderLabels(r.columns)
        for i in range(len(r.index)):
            for j in range(len(r.columns)):
                tbr.setItem(i,j,widgets.QTableWidgetItem(str(r.iat[i, j])))

        if self.blits_data.results_acquired['fractional saturation']:
            p = self.blits_data.get_fractional_saturation_params_dataframe()
            tbp = self.tblFitParams
            tbp.setColumnCount(len(p.columns))
            tbp.setRowCount(len(p.index))
            tbp.setVerticalHeaderLabels(p.index)
            tbp.setHorizontalHeaderLabels(p.columns)
            for i in range(len(p.index)):
                for j in range(len(p.columns)):
                    tbp.setItem(i,j,widgets.QTableWidgetItem(str(p.iat[i, j])))
            
            f = self.blits_data.get_fractional_saturation_curve()
            tbf = self.tblFittedCurve
            tbf.setColumnCount(len(f.columns))
            tbf.setRowCount(len(f.index))
            tbf.setHorizontalHeaderLabels(f.columns)
            for i in range(len(f.index)):
                for j in range(len(f.columns)):
                    tbf.setItem(i,j,widgets.QTableWidgetItem(str(f.iat[i, j])))
            
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
        
# Standard main loop code
if __name__ == '__main__':
    import sys
#    sys.tracbacklimit = 10
    app = widgets.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
