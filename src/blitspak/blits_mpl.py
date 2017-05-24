'''
Created on 23 May 2017

@author: schilsm
'''
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:43:23 2016

@author: Maria Schilstra
"""
#from PyQt5 import QtGui as gui
#from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets

from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT)


class MplCanvas(FigureCanvas):
    """ 
    Class representing the FigureCanvas widget to be embedded in the GUI
    """
    curve_colours = ['blue',
                     'green',
                     'red',
                     'orange',
                     'cyan',
                     'magenta',
                     'purple',
                     'brown',
                     'white',
                     'black'
                    ] 

    def __init__(self, parent):
        self.fig = Figure()
        
        self.gs = gridspec.GridSpec(10, 2) 
        self.gs.update(left=0.07, right=0.97, top=0.97, 
                       bottom=0.07, hspace=0.5)
        self.data_plot = self.fig.add_subplot(self.gs[2:,0])
        self.data_res_plot = self.fig.add_subplot(self.gs[0:2,0], 
                                                  sharex=self.data_plot)
        self.load_plot = self.fig.add_subplot(self.gs[:3,1])
        self.fsat_plot = self.fig.add_subplot(self.gs[6:,1])
        self.fsat_res_plot = self.fig.add_subplot(self.gs[4:6,1], 
                                                  sharex=self.fsat_plot)
        self.fig.subplots_adjust(bottom=0.2)
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, 
                        widgets.QSizePolicy.Expanding, 
                        widgets.QSizePolicy.Expanding)
        self.set_fig_annotations()
        FigureCanvas.updateGeometry(self)  

    def set_fig_annotations(self):
        self.data_plot.set_xlabel("Time (s)")
        self.data_plot.set_ylabel("Response (nm)")
        self.data_res_plot.set_ylabel("Residuals (1)")
        self.load_plot.set_xlabel("Trace name")
        self.load_plot.set_ylabel("Sugar loading")
        self.fsat_plot.set_xlabel("Sugar loading")
        self.fsat_plot.set_ylabel("Fractional saturation")
        self.fsat_res_plot.set_ylabel("Residuals (2)")
        self.data_res_plot.locator_params(axis='y',nbins=4)        
        self.fsat_res_plot.locator_params(axis='y',nbins=4)        
    
        
    def draw_data(self, x, y, boundaries = None):
        self.data_plot.cla()
        self.data_res_plot.cla()
        dp = self.data_plot.plot(x, y)
        for i in range(len(dp)):
            dp[i].set_color(self.curve_colours[i])
        self.set_fig_annotations()
        if not boundaries is None:
            self.boundary_markers = []
            for pos in boundaries:
                self.boundary_markers.append(self.createVLine(pos=pos))
        self.fig.canvas.draw()
        
    def draw_fitted_data(self, x, y):
        self.data_plot.plot(x, y, color='k',linestyle='--')
        self.set_fig_annotations()
        self.fig.canvas.draw()

    def draw_residuals(self, x, y):
        rp = self.data_res_plot.plot(x, y)
        for i in range(len(rp)):
            rp[i].set_color(self.curve_colours[i])
        self.set_fig_annotations()
        self.fig.canvas.draw()

    def draw_sugar_loading(self, load):
        self.load_plot.cla()
        load.plot.bar(ax=self.load_plot,color=self.curve_colours)
        self.set_fig_annotations()
        self.fig.canvas.draw()
        
    def draw_fractional_saturation(self, x_obs, y_obs, y_res, x_fit, y_fit, params):
        self.fsat_plot.cla()
        self.fsat_res_plot.cla()
        y_max, x_half, h = params['ymax'], params['xhalf'], params['h']
        y_obs_norm, y_res_norm, y_fit_norm = y_obs/y_max, y_res/y_max, y_fit/y_max
        self.fsat_plot.plot(x_fit, y_fit_norm, color = 'k')
        self.fsat_plot.plot(x_obs, y_obs_norm, linestyle='None', marker='o', 
                                markerfacecolor='r', markersize=12)
        txtparams = 'Max: {:.1f} \nHalf sat: {:.2f} \nHill coeff: {:.1f}'.format(y_max, x_half, h)
        self.fsat_plot.text(x_half, 0.1, txtparams, fontsize=12)
        self.fsat_res_plot.plot(x_obs, y_res_norm, linestyle='None', marker='o', 
                                markerfacecolor='r', markersize=6)
        self.set_fig_annotations()
        self.fig.canvas.draw()
        
        
    def createVLine(self, pos=0.0, color='k', width=1.0):
        vLine = self.data_plot.axvline(pos)
        return vLine
        
    def moveVLine(self, vLine, pos):
        if not vLine is None:
            vLine.set_xdata(pos)
            self.fig.canvas.draw()
        
    def removeVLine(self, vLine):
        if not vLine is None:
            vLine.remove()
            del vLine
                
    def clear_figure(self):
        self.data_plot.cla()
        self.data_res_plot.cla()
        self.fsat_plot.cla()
        self.fsat_res_plot.cla()
        self.load_plot.cla()
        self.set_fig_annotations()
        self.fig.canvas.draw()
        
        
class NavigationToolbar(NavigationToolbar2QT):
                        
    def __init__(self, canvas_, parent_):
        NavigationToolbar2QT.__init__(self,canvas_,parent_)
        
    def switch_off_pan_zoom(self):
        if self._active == "PAN":
            self.pan()
        elif self._active == "ZOOM":
            self.zoom()
