"""
Created on 23 May 2017

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
        
        self.gs = gridspec.GridSpec(10, 1) 
        self.gs.update(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=1.5)
        self.data_plot = self.fig.add_subplot(self.gs[2:,:])
        self.data_res_plot = self.fig.add_subplot(self.gs[0:2,:], sharex=self.data_plot)
                
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, widgets.QSizePolicy.Expanding, widgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self) 
        self.curves = {}

    def set_fig_annotations(self):
        self.data_plot.set_xlabel("Time (s)")
        self.data_plot.set_ylabel("Response (nm)")
        self.data_res_plot.set_ylabel("Residuals")
        self.data_res_plot.locator_params(axis='y',nbins=4)        
    
    def draw_data(self, x, y, boundaries = None):
        self.data_plot.cla()
        self.data_res_plot.cla()
        dp = self.data_plot.plot(x, y)
        for i in range(len(dp)):
            dp[i].set_color(self.curve_colours[i])
        self.set_fig_annotations()
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
 
    def clear_figure(self):
        self.data_plot.cla()
        self.data_res_plot.cla()
        self.set_fig_annotations()
        self.fig.canvas.draw()
        
        
class NavigationToolbar(NavigationToolbar2QT):
                        
    def __init__(self, canvas_, parent_):
        self.toolitems = tuple([t for t in NavigationToolbar2QT.toolitems if
                 t[0] in ('Home', 'Back', 'Forward', 'Pan', 'Zoom', 'Save')])
        NavigationToolbar2QT.__init__(self,canvas_,parent_)  
        
    def switch_off_pan_zoom(self):
        if self._active == "PAN":
            self.pan()
        elif self._active == "ZOOM":
            self.zoom()
