## Copyright (C) 2016
## Written by Travis Thieme <tthieme@hawaii.edu>
## University of Hawai`i at Hilo
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

##########################
## Load required modules##
##########################

import numpy as np
#import scipy as sp
#import sympy as sy
#import statsmodels as stats
#import astropy as astro
#import math as m
#import matplotlib.pyplot as plt
#import pyqtgraph as pyqt
#import wxpython as wxp

#import pandas
#import random
import time 
#import pdb
#from scipy.optimize import fsolve, root, findRoot
from mayavi import mlab

## ###################################
##
## Author: Travis Thieme
## Created on: June 21th, 2016
##
## Application Name: Plotting and Fitting
## Programming Language: Python
## 
## Description: This program 
## 
## ###################################

# Project Info
name = 'Travis Thieme'
date  = time.strftime("%d/%m/%Y")
clock = time.strftime("%H:%M:%S %Z")

def project_info(project = ''):

    """
    This function prints the info for the project your working on. Info such as author,
    date, current time, and the project name.
    """
    
    name = 'Travis Thieme'
    date = time.strftime("%A, %d/%m/%Y")
    clock = time.strftime("%H:%M:%S %Z")
    project_name = project

    print name
    print date
    print clock
    print project_name

def mayavi_demo():
  
    """
    This function is a practice for the mayavi package in python
    """
    # function
    x = np.linspace(-10,10,100)
    y = np.sin(x)
    z = np.zeros(100)
    
    # mayavi plot
    mlab.figure(1,bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    mlab.plot3d(x,y,z)
    mlab.axes(extent=[-10,10,-10,10,-10,10], \
                  x_axis_visibility=False, \
                  y_axis_visibility=False, \
                  z_axis_visibility=False)
    mlab.move(1,0,0)
    mlab.screenshot()
    #plt.show(img)    
    mlab.show()
    
def mayavi_demo2():
  
    """
    This function is a practice for the mayavi package in python
    """
    # function
    x = np.linspace(-10,10,100)
    y = np.linspace(-10,10,100)
    z = x*np.sin(x)*np.sin(y)
    
    # mayavi plot
    mlab.figure(1,bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    mlab.contour3d(x,y,z)
    mlab.axes(extent=[-10,10,-10,10,-10,10], \
                  x_axis_visibility=False, \
                  y_axis_visibility=False, \
                  z_axis_visibility=False)
    mlab.move(1,0,0)  
    mlab.show()