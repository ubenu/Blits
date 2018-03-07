"""
Created on 23 May 2017

@author: Maria Schilstra
"""
#from PyQt5 import QtGui as gui
#from PyQt5 import QtCore as qt
import numpy as np
from PyQt5 import QtWidgets as widgets

from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT)


class MplCanvas(FigureCanvas):
    """ 
    Class representing the FigureCanvas widget to be embedded in the GUI
    """
    colour_seq = ['blue',
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
        self.gs.update(left=0.15, right=0.95, top=0.95, bottom=0.1, hspace=5.0)
        self.data_plot = self.fig.add_subplot(self.gs[2:,:])
        self.data_res_plot = self.fig.add_subplot(self.gs[0:2,:], sharex=self.data_plot)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, widgets.QSizePolicy.Preferred, widgets.QSizePolicy.Preferred)
        FigureCanvas.updateGeometry(self) 
        
        self.dplot_xlims, self.dplot_ylims = None, None
        self.curve_colours = {}
        self.remove_vlines() # sets the vertical lines to None
        
    def on_move(self):
        pass

    def set_fig_annotations(self, ylabel="Value", rlabel="Residuals"):
        self.data_plot.set_ylabel(ylabel)
        self.data_res_plot.set_ylabel(rlabel)
        self.data_res_plot.locator_params(axis='y',nbins=4)
        
    def set_colours(self, series_names):
        self.curve_colours = {}
        for name in series_names:
            i = series_names.index(name) % len(self.colour_seq)
            self.curve_colours[name] = self.colour_seq[i]
            
    def get_series_colour(self, series_name):
        if self.series_in_plot(series_name):
            return self.curve_colours[series_name]
        return ''
    
    def series_in_plot(self, series_name):
        return series_name in self.curve_colours
    
    def has_vertical_lines(self):
        return self.vline0 is not None and self.vline1 is not None
    
    def clear_plots(self):
        self.data_plot.cla()
        self.data_res_plot.cla()
        self.set_fig_annotations()
        self.dplot_xlims, self.dplot_ylims = None, None
        self.fig.canvas.draw()
    
    def draw_series(self, series_name, x, y, kind):
        """
        Draw a single curve.
        @series_name: series id (string, must be unique)
        @x: x-axis values (pandas series)
        @y: y-axis values (pandas series)
        @kind: 'primary', 'calculated', 'residuals'
        """
        xdif = np.mean(np.diff(x))
        xspan = np.max(x) - np.min(x)
        if kind in ('primary', 'residuals'):
            marker = 'o'
            if xdif != 0.0:
                if xspan / xdif > 50:  
                    marker = '-'
        elif kind == 'calculated':
            marker = '--'
        if not self.series_in_plot(series_name):
            i = len(self.curve_colours.keys()) % len(self.colour_seq)
            self.curve_colours[series_name] = self.colour_seq[i]
        if kind in ('primary', 'calculated'):
            self.data_plot.plot(x, y, marker, color=self.curve_colours[series_name])
            if kind == 'primary':
                self.dplot_xlims, self.dplot_ylims = self.data_plot.get_xlim(), self.data_plot.get_ylim()
            if self.dplot_xlims is not None:
                self.data_plot.set_xlim(self.dplot_xlims)
            if self.dplot_ylims is not None:
                self.data_plot.set_ylim(self.dplot_ylims)                
            self.data_plot.ticklabel_format(style='sci', scilimits=(-3,3), axis='both')
        if kind == 'residuals':
            self.data_res_plot.plot(x, y, color=self.curve_colours[series_name])
                     
        self.fig.canvas.draw()
               
 
    def set_vlines(self, x_limits, x_outer_limits):
        self.vline0 = DraggableLine(self.data_plot.axvline(x_limits[0],
                                                           lw=1, 
                                                           ls='--', 
                                                           color='k'), 
                                    x_outer_limits)
        self.vline1 = DraggableLine(self.data_plot.axvline(x_limits[1],
                                                           lw=1, 
                                                           ls='--', 
                                                           color='k'), 
                                    x_outer_limits)           

    def remove_vlines(self):
        self.vline0 = None
        self.vline1 = None
        
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
            
class DraggableLine:
    """
    Based on DraggableRectangle exercise in https://matplotlib.org/users/event_handling.html
    """
    def __init__(self, line, xlims):
        self.line = line
        self.vline_xlims = xlims
        self.connect()
        self.press = None
        
    def get_x(self):
        return self.line.get_xdata()[0]
    
    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.line.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.line.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.line.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes == self.line.axes: 
            contained = self.line.contains(event)[0]
            if contained:
                self.press = event.xdata 
        return
        
    def on_motion(self, event):
        'on motion we will move the line if the mouse is over us'
        if self.press is None: 
            return
        if event.inaxes != self.line.axes: 
            return
        if event.xdata < self.vline_xlims[0] or event.xdata > self.vline_xlims[1]:
            return
        newx = np.ones_like(self.line.get_xdata()) * event.xdata
        self.line.set_xdata(newx)
        self.line.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        self.line.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.line.figure.canvas.mpl_disconnect(self.cidpress)
        self.line.figure.canvas.mpl_disconnect(self.cidrelease)
        self.line.figure.canvas.mpl_disconnect(self.cidmotion)
