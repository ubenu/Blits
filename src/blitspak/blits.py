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

import pandas as pd, numpy as np, copy as cp


from matplotlib.widgets import SpanSelector
from blitspak.blits_mpl import MplCanvas, NavigationToolbar
from blitspak.blits_data import BlitsData
from blitspak.scrutinize_dialog import ScrutinizeDialog
from blitspak.function_dialog import FunctionSelectionDialog
from blitspak.data_creation_dialog import DataCreationDialog
from blitspak.crux_table_model import CruxTableModel
from functions.framework import FunctionsFramework

#import blitspak.blits_ui as ui
from PyQt5.uic import loadUiType
from win32com.test.testAXScript import AXScript
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
        self.grp_show_axis = widgets.QGroupBox()
        self.axis_layout = widgets.QHBoxLayout()
        self.grp_show_axis.setLayout(self.axis_layout)
        self.grp_show_axis.setSizePolicy(widgets.QSizePolicy.Maximum, widgets.QSizePolicy.Maximum)
        self.axisgrp_layout = widgets.QHBoxLayout()
        self.axisgrp_layout.addWidget(self.grp_show_axis)
        self.mpl_layout.addLayout(self.axisgrp_layout)
        self.mpl_layout.addWidget(self.plot_toolbar)
        
        self.action_open.triggered.connect(self.on_open)
        self.action_create.triggered.connect(self.on_create)
        self.action_close.triggered.connect(self.on_close_data)
        self.action_save.triggered.connect(self.on_save)
        self.action_select_function.triggered.connect(self.on_select_function)
        self.action_analyze.triggered.connect(self.on_analyze)
        self.action_quit.triggered.connect(self.close) 
        self.action_apply.triggered.connect(self.on_apply_current) 
        self.action_estimate.triggered.connect(self.on_estimate) 
        
        self.chk_global.clicked.connect(self.on_global)
        self.chk_subsection.clicked.connect(self.on_subsection)
        
        ft = gui.QFont('Calibri', 14)
        self.btn_est = widgets.QPushButton("Estimate")
        self.btn_est.setFont(ft)
        self.btn_est.clicked.connect(self.on_estimate)
        self.btn_apply = widgets.QPushButton("Apply")
        self.btn_apply.setFont(ft)
        self.btn_apply.clicked.connect(self.on_apply_current)
        self.btn_fit = widgets.QPushButton("Perform fit")
        self.btn_fit.setFont(ft)
        self.btn_fit.clicked.connect(self.on_analyze)
        self.bbox_fit.addButton(self.btn_fit, widgets.QDialogButtonBox.ActionRole)
        self.bbox_fit.addButton(self.btn_apply, widgets.QDialogButtonBox.ActionRole)
        self.bbox_fit.addButton(self.btn_est, widgets.QDialogButtonBox.ActionRole)

#         self.span = SpanSelector(self.canvas.data_plot, 
#                                  self.on_select_span, 
#                                  'horizontal', 
#                                  useblit=True, 
#                                  rectprops=dict(alpha=0.5, facecolor='red')
#                                  )
        
        self.blits_data = BlitsData()
        self.blits_fitted = BlitsData()
        self.blits_residuals = BlitsData()
        self.file_name = ""
        self.file_path = ""
        self.phase_number = 0
        self.phase_name = "Phase"
        self.phase_list = []
        self.axis_selector_buttons = None
        self.current_function = None
        self.axes_limits = None
        
        self.current_state = self.START        
        self.update_ui()
        
    def on_analyze(self):
        if self.current_state in (self.READY_FOR_FITTING, self.FITTED):
            fitted_params, sigmas, confidence_intervals, tol = self.perform_fit()
            self.make_crux_curves(fitted_params, 100)
            self.draw_current_data_set()
            self.write_param_values_to_table(fitted_params)
            self.current_state = self.FITTED
            self.update_ui()
        pass
            
    def on_apply_current(self):
        if self.current_state in (self.READY_FOR_FITTING, self.FITTED):
            params = self.get_param_values_from_table(self.get_selected_series_names())
            self.make_crux_curves(params, 100)
            self.draw_current_data_set()
        pass  
    
    def on_close_data(self):
        if self.current_state in (self.DATA_ONLY, self.READY_FOR_FITTING, self.FITTED, ):
            self.current_xaxis = None
            self.set_axis_selector()
            self.canvas.clear_figure()
            
            self.blits_data = BlitsData()
            self.blits_fitted = BlitsData()
            self.blits_residuals = BlitsData()
            
            if self.current_state == self.DATA_ONLY:
                self.current_state = self.START
            else:
                self.current_state = self.FUNCTION_ONLY
                
            self.set_current_function_ui()
            self.update_ui()
        pass
            
    def on_create(self):  
        if self.current_state in (self.FUNCTION_ONLY, ):
            self.create_data_set_dialog = DataCreationDialog(None, self.current_function)
            if self.create_data_set_dialog.exec() == widgets.QDialog.Accepted:
                template = self.create_data_set_dialog.template
                self.set_params_view(template[1])
                self.blits_data.create_working_data_from_template(template)
                self.current_state = self.READY_FOR_FITTING
                self.current_xaxis = self.blits_data.independent_names[0]
                self.set_axis_selector()
                self.update_ui()
                
    def on_estimate(self):
        if self.current_state in (self.READY_FOR_FITTING, self.FITTED):
            fn_p0 = self.current_function.p0
            n_par = len(self.current_function.parameters)
            data = self.get_data_for_fitting(self.get_selected_series_names())
            ffw = FunctionsFramework()
            params = ffw.get_initial_param_estimates(data, fn_p0, n_par)
            self.write_param_values_to_table(params)
#            self.make_crux_curves(params, 100)
            self.draw_current_data_set()
        pass 
    
    def on_global(self):
        pass          
    
    def on_open(self):
        if self.current_state in (self.START, self.FUNCTION_ONLY, ):
            file_path = widgets.QFileDialog.getOpenFileName(self, 
            "Open Data File", "", "CSV data files (*.csv);;All files (*.*)")[0]
            if file_path:
                self.file_path = file_path
                info = qt.QFileInfo(file_path)
                self.blits_data.file_name = info.fileName()
                self.blits_data.import_data(file_path)
                axes = self.blits_data.independent_names
                self.current_xaxis = self.blits_data.independent_names[0]
                if self.current_state == self.START:
                    self.current_state = self.DATA_ONLY
                else:
                    if len(self.current_function.independents) <= len(axes):
                        self.current_state = self.READY_FOR_FITTING
                    else:
                        self.current_function = None
                        self.current_state = self.DATA_ONLY
                        
                self.axes_limits = pd.DataFrame(index=axes, columns=['subsection', 'inner', 'outer'])
                mins, maxs = self.blits_data.series_extremes()
                for x in axes:                    
                    limits = (mins.loc[:, x].min(),
                              maxs.loc[:, x].max())
                    self.axes_limits.loc[x, 'subsection'] = False
                    self.axes_limits.loc[x, 'inner'] = limits
                    self.axes_limits.loc[x, 'outer'] = limits

                self.set_axis_selector()
                self.set_current_function_ui()

                self.update_ui()
                
    def on_save(self):
        file_path = ""
        if self.current_state in (self.DATA_ONLY, self.READY_FOR_FITTING, ):
            file_path = widgets.QFileDialog.getSaveFileName(self, 
            "Save data", "", "CSV data files (*.csv);;All files (*.*)")[0]
        if self.current_state in (self.FITTED, ):
            file_path = widgets.QFileDialog.getSaveFileName(self, 
            "Save results", "", "CSV data files (*.csv);;All files (*.*)")[0]
        if file_path:
            pass
#             self.blits_data.export_results(file_path)
            
    def on_select_function(self):
        if self.current_state in range(self.NSTATES):  # should work from all states
            name, n_axes = "", np.inf
            if not self.current_state in (self.START, self.DATA_ONLY):  # a current function exists
                name = self.current_function.name
            if self.current_state in (self.DATA_ONLY, self.READY_FOR_FITTING, self.FITTED):
                n_axes = len(self.blits_data.independent_names)
            self.function_dialog = FunctionSelectionDialog(self, n_axes=n_axes, selected_fn_name=name)
            if self.function_dialog.exec() == widgets.QDialog.Accepted:
                self.current_function = self.function_dialog.get_selected_function()
                self.blits_fitted = BlitsData()
                self.blits_residuals = BlitsData()
                if self.current_state in (self.START, self.FUNCTION_ONLY):
                    self.current_state = self.FUNCTION_ONLY
                else:
                    self.current_state = self.READY_FOR_FITTING
                self.draw_current_data_set()
                print(self.axes_limits)
                self.set_current_function_ui()
                self.update_ui()    


    def on_subsection(self):    
        if self.blits_data.has_data():
            if self.chk_subsection.isChecked():
                # user has toggled the checkbox to checked:
                mins, maxs = self.blits_data.series_extremes()
                x_outer_limits = (mins.loc[:, self.current_xaxis].min(), 
                                  maxs.loc[:, self.current_xaxis].max())
                x_limits = cp.deepcopy(x_outer_limits)
                self.canvas.set_vlines(x_limits, x_outer_limits)
            else:
                self.canvas.remove_vlines()
            self.preserve_vlines()
            self.draw_current_data_set()
            print(self.axes_limits)
        pass        
    
    def preserve_vlines(self):
        if self.blits_data.has_data():
            mins, maxs = self.blits_data.series_extremes()
            x_outer_limits = (mins.loc[:, self.current_xaxis].min(), 
                              maxs.loc[:, self.current_xaxis].max())
            x_limits = cp.deepcopy(x_outer_limits)
            self.axes_limits.loc[self.current_xaxis, 'subsection'] = False
            if self.canvas.has_vertical_lines():
                x_limits = (self.canvas.vline0.get_x(), 
                            self.canvas.vline1.get_x())
                self.axes_limits.loc[self.current_xaxis, 'subsection'] = True
            self.axes_limits.loc[self.current_xaxis, 'inner'] = x_limits
            self.axes_limits.loc[self.current_xaxis, 'outer'] = x_outer_limits
            print(self.axes_limits)
        else:
            self.axes_limits = None        

    def on_xaxis_state_changed(self, checked):
        btn = self.sender()
        xaxis = btn.text()
        if btn.isChecked():
            self.current_xaxis = xaxis
            self.draw_current_data_set()   
            
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

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                self.clearLayout(child.layout())            
                                    
    def get_constant_params_from_table(self, series_names):
        """
        Returns an (n_curves, n_params)-shaped array of Boolean values 
        (with rows and columns parallel to self.series_names and self.current_function.parameters, 
        respectively) with values for each parameter for each series); if True, 
        parameter values is constant, if False, parameter value is variable.
        """
        return cp.deepcopy(self.parameters_model.df_checks)[series_names].as_matrix().transpose()
        
    def get_data_for_fitting(self, series_names):
        data = []
        mins, maxs = self.blits_data.series_extremes()
        x_outer_limits = (mins.loc[:, self.current_xaxis].min(), 
                          maxs.loc[:, self.current_xaxis].max())
        start, stop = x_outer_limits
        if self.canvas.has_vertical_lines():
            start, stop = self.canvas.vline0.get_x(), self.canvas.vline1.get_x()
            if start > stop: 
                temp = start
                start = stop
                stop = temp
        for s in series_names:
            series = self.blits_data.series_dict[s] # the full data set
            indmin, indmax = np.searchsorted(series[self.current_xaxis],(start, stop))
            selection = cp.deepcopy(series[indmin:indmax]).as_matrix().transpose()
            if len(data) == 0:
                data = [selection]
            else:
                data.append(selection)
        return data
        
    def get_param_values_from_table(self, series_names):
        """
        Returns an (n_curves, n_params)-shaped array (with rows and columns 
        parallel to self.series_names and self.current_function.parameters, 
        respectively) with values for each parameter for each series).  
        """
        return cp.deepcopy(self.parameters_model.df_data)[series_names].as_matrix().transpose()
    
    def get_selected_series_names(self):
        cols = cp.deepcopy(self.parameters_model.df_data.columns)
        return cols.tolist()

    def get_linked_params_from_table(self, series_names):
        """
        STILL TO BE IMPLEMENTED
        
        Returns an (n_curves, n_params)-shaped array (with rows and columns parallel to 
        self.series_names and self.current_function.parameters, respectively)
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
        shape = self.parameters_model.df_data.shape
        
        links_array = np.arange(shape[0] * shape[1]).reshape((shape[0], shape[1])).transpose()
        return links_array
         
    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False  
        
    def line_icon(self, color):
        pixmap = gui.QPixmap(50,10)
        pixmap.fill(gui.QColor(color))
        icon = gui.QIcon(pixmap)
        return icon  
    
    def make_crux_curves(self, params, n_points):
        mins, maxs = self.blits_data.series_extremes()
        # mins, maxs: index: series names, columns: independent names + y (dependent)
        series_names = mins.index
        axes_names = mins.columns
        min_xs, max_xs = mins.iloc[:, :-1].as_matrix(), maxs.iloc[:, :-1].as_matrix()
        series_dict = {}
        for series_name, xmin, xmax, series_params in zip(series_names, min_xs, max_xs, params):
            df_data = pd.DataFrame(index=[], columns=range(n_points))
            # create values for the independent axes (shape (n_independents, n_points))
            for v_start, v_end in zip(xmin, xmax):
                x = pd.DataFrame(np.linspace(v_start, v_end, n_points)).transpose()
                df_data = pd.concat((df_data, x))
            x = df_data.as_matrix()
            # create the y values and put them in a DataFrame, transpose for easy concatenation
            y = pd.DataFrame(self.current_function.func(x, series_params)).transpose()
            df_data = pd.concat((df_data, y))
            l_axes_names = axes_names.tolist()[:-1]
            l_axes_names.append(series_name)
            df_data.index = l_axes_names
            series_dict[series_name] = df_data.transpose()
        self.blits_fitted = BlitsData()
        self.blits_fitted.series_names = np.array(series_names.tolist())
        self.blits_fitted.independent_names = np.array(l_axes_names)[:-1]
        self.blits_fitted.series_dict = series_dict
                       
    def perform_fit(self):
        func = self.current_function.func
        series_names = self.get_selected_series_names()
        data = self.get_data_for_fitting(series_names)
        param_values = self.get_param_values_from_table(series_names)
        const_params = self.get_constant_params_from_table(series_names)             
        links = self.get_linked_params_from_table(series_names)
        fitted_params = cp.deepcopy(param_values)
        sigmas = np.empty_like(fitted_params)
        confidence_intervals = np.empty_like(fitted_params)
        tol = None
        results = None  
        ffw = FunctionsFramework()
        if self.chk_global.checkState() == qt.Qt.Checked:
            results = ffw.perform_global_curve_fit(data, func, param_values, const_params, links)
            fitted_params = results[0]
            sigmas = results[1]
            confidence_intervals = results[2]
            tol = results[3]
        else:
            tol = []
            n = 0
            for d, p, c, l in zip(data, param_values, const_params, links):
                d = [d, ]
                p = np.reshape(p, (1, p.shape[0]))
                c = np.reshape(c, (1, c.shape[0]))
                l = np.reshape(l, (1, l.shape[0]))
                results = ffw.perform_global_curve_fit(d, func, p, c, l)
                fitted_params[n] = results[0]
                sigmas[n] = results[1]
                confidence_intervals[n] = results[2]
                tol.append(results[3])
                n += 1
        return fitted_params, sigmas, confidence_intervals, tol      
            
    def set_axis_selector(self):
        self.axis_selector_buttons = {}
        self.clearLayout(self.axis_layout)
        if self.blits_data.has_data():
            self.axis_layout.addStretch()
            for name in self.blits_data.independent_names:
                btn = widgets.QRadioButton()
                btn.setText(name)
                btn.toggled.connect(self.on_xaxis_state_changed)
                self.axis_layout.addWidget(btn)
                self.axis_selector_buttons[btn.text()] = btn
            self.axis_layout.addStretch()  
            if not self.current_xaxis is None:
                self.axis_selector_buttons[self.current_xaxis].setChecked(True)
            
    def set_current_function_ui(self):        
        if self.current_state in (self.DATA_ONLY, self.START, ): # there is no self.current_functions
            self.parameters_model = None
            self.tbl_params.setModel(None)
        else:
            indx_pars = self.current_function.parameters
            cols_pars = [] 
            if self.current_state in (self.READY_FOR_FITTING, self.FITTED):
                cols_pars = self.blits_data.series_names
            df_pars = pd.DataFrame(np.ones((len(indx_pars), len(cols_pars)), dtype=float), index=indx_pars, columns=cols_pars) 
            self.set_params_view(df_pars)                   
            self.lbl_fn_name.setText("Selected function: " + self.current_function.name)
            self.txt_description.setText(self.current_function.long_description)
                
    def set_params_view(self, df_pars, checkable=None):
        self.tbl_params.setModel(None)
        if checkable is None:
            checkable = list(range(len(df_pars.columns)))
        self.parameters_model = CruxTableModel(df_pars, checkable)
        self.tbl_params.setModel(self.parameters_model)
        self.tbl_params.setSizeAdjustPolicy(widgets.QAbstractScrollArea.AdjustToContents)
                
    def draw_current_data_set(self):
        self.canvas.clear_plots() 
        if self.blits_data.has_data():
            self.canvas.set_colours(self.blits_data.series_names.tolist())
            for key in self.blits_data.series_names:
                series = self.blits_data.series_dict[key]
                x = series[self.current_xaxis] 
                y = series[key] 
                self.canvas.draw_series(key, x, y, 'primary')
        if self.blits_fitted.has_data():
            for key in self.blits_fitted.series_names:
                series = self.blits_fitted.series_dict[key]
                x = series[self.current_xaxis] 
                y = series[key] 
                self.canvas.draw_series(key, x, y, 'calculated')
        if self.blits_residuals.has_data():
            for key in self.blits_residuals.series_names:
                series = self.blits_residuals.series_dict[key]
                x = series[self.current_xaxis] 
                y = series[key] 
                self.canvas.draw_series(key, x, y, 'residuals')

    def write_param_values_to_table(self, param_values):
        self.parameters_model.change_content(param_values.transpose())
        #self.parameters_model.df_data[:] = param_values.transpose()
        #self.tbl_params.resizeColumnsToContents() # This redraws the table (necessary)
            
    def update_ui(self):
        if self.current_state == self.START:
            self.action_open.setEnabled(True)
            self.action_create.setEnabled(False)
            self.action_close.setEnabled(False)
            self.action_save.setEnabled(False)
            self.action_select_function.setEnabled(True)
            self.action_analyze.setEnabled(False)
            self.btn_apply.setEnabled(False)
            self.btn_fit.setEnabled(False)
            self.btn_est.setEnabled(False)
            self.action_quit.setEnabled(True)
            self.chk_subsection.setEnabled(False)   
            self.chk_global.setEnabled(False) 
            self.chk_subsection.setChecked(False)
            self.chk_global.setChecked(False) 
        elif self.current_state == self.DATA_ONLY:
            self.action_open.setEnabled(False)
            self.action_create.setEnabled(False)
            self.action_close.setEnabled(True)
            self.action_save.setEnabled(True)
            self.action_select_function.setEnabled(True)
            self.action_analyze.setEnabled(False)
            self.btn_apply.setEnabled(False)
            self.btn_fit.setEnabled(False)
            self.btn_est.setEnabled(False)
            self.action_quit.setEnabled(True) 
            self.chk_subsection.setEnabled(False)   
            self.chk_global.setEnabled(False)  
            self.chk_global.setChecked(False) 
        elif self.current_state == self.FUNCTION_ONLY:
            self.action_open.setEnabled(True)
            self.action_create.setEnabled(True)
            self.action_close.setEnabled(False)
            self.action_save.setEnabled(False)
            self.action_select_function.setEnabled(True)
            self.action_analyze.setEnabled(False)
            self.btn_apply.setEnabled(False)
            self.btn_fit.setEnabled(False)
            self.btn_est.setEnabled(False)
            self.action_quit.setEnabled(True)     
            self.chk_subsection.setEnabled(False)   
            self.chk_global.setEnabled(False)  
            self.chk_subsection.setChecked(False)
            self.chk_global.setChecked(False) 
        elif self.current_state == self.READY_FOR_FITTING:
            self.action_open.setEnabled(False)
            self.action_create.setEnabled(False)
            self.action_close.setEnabled(True)
            self.action_save.setEnabled(True)
            self.action_select_function.setEnabled(True)
            self.action_analyze.setEnabled(True)
            self.btn_apply.setEnabled(True)
            self.btn_fit.setEnabled(True)
            self.btn_est.setEnabled(True)
            self.action_quit.setEnabled(True)     
            self.chk_subsection.setEnabled(True)   
            self.chk_global.setEnabled(True)  
        elif self.current_state == self.FITTED:
            self.action_open.setEnabled(False)
            self.action_create.setEnabled(False)
            self.action_close.setEnabled(True)
            self.action_save.setEnabled(True)
            self.action_select_function.setEnabled(True)
            self.action_analyze.setEnabled(True)
            self.btn_apply.setEnabled(True)
            self.btn_fit.setEnabled(True)
            self.btn_est.setEnabled(True)
            self.action_quit.setEnabled(True)     
            self.chk_subsection.setEnabled(True)   
            self.chk_global.setEnabled(True)  
        else:
            print('Illegal state')
                                          


# Standard main loop code
if __name__ == '__main__':
    import sys
#    sys.tracbacklimit = 10
    app = widgets.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())

#     def create_results_tab(self, phase_id, model_string, results_table):
#         new_tab = widgets.QWidget()
#         lo = widgets.QVBoxLayout(new_tab)
#         x = widgets.QLabel(model_string)
#         lo.addWidget(x)
#         lo.addWidget(results_table)
#         new_tab.setLayout(lo)
#         self.tabWidget.addTab(new_tab, phase_id)

#     def on_select_span(self, xmin, xmax):
#         self.span.set_active(False)
#         if xmin != xmax:
#             mins, maxs = self.blits_data.series_extremes()
#             # mins, maxs: index: series names, columns: independent names + y (dependent)  
#             x_outer_limits = (mins.loc[:, self.current_xaxis].min(), maxs.loc[:, self.current_xaxis].max())
#             x_limits = (xmin, xmax)                
#             if xmin > xmax:
#                 x_limits = (xmax, xmin)
#             self.canvas.set_vlines(x_limits, x_outer_limits)





