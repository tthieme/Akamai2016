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
import scipy as sp
#import sympy as sy
#import statsmodels as stats
#import astropy as astro
#import math as m
import matplotlib.pyplot as plt
#import pyqtgraph as pyqt
#import wxpython as wxp

#import pandas
#import random
import time 
#import pdb

#from scipy.optimize import fsolve, root, findRoot

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

# Global Variables
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
    
    
def plot_bessel():
    
    """
    This function plots the J1 bessel function.
    """
    
    # set interval
    start = 0
    end = 20
    
    # set values for distribution
    A = 1
    b = 3
    
    # linearly spaced numbers from -10 to 10 with 100 increments
    x = np.linspace(start,end,100)
    y = bessel_j1(x, A, b)
    
    # find first three bessel roots
    a_m = (sp.special.jn_zeros(1, 3))/b
    print a_m[0]
    
    plt.clf() # clear plot
    plt.title('Bessel Plot') # plot title
    plt.figtext(0.01,0.95,'Created by: ' + name) # Add created by to top left corner
    plt.figtext(0.72,0.95, 'Todays Date: '  + date) # Add date to top right corner
    plt.plot(x, np.zeros(len(x)), 'k')    
    plt.plot(x,y, 'b', label = '$J_1$') # plot origional
    plt.xlabel('x') # label x axis
    plt.ylabel('y') # label y axis
    plt.legend() # show legend
    plt.grid(True) # show grid lines
    plt.show() # show plot
  
def bessel_j1(x, A, b):  

    """
    This function calculates the J1 bessel function.
    """
    
    return A*sp.special.j1(b*x)
    
def mag_field():
  
    """
    This function 
    """
    
    ###FIRST###
    
    # bessel scaling
    #v_scale = 1
    h_scale = 1
    
    # model parameters
    R = 3.8918
    k = [] # hold calculated k components
    k.append(0.9549) # k_1
    k.append(0.4608) # k_2
    k.append(0.6320) # k_3
    h = 0.3250
    B_0 = 3.3118 
    
    # intervals for r and z values
    r = np.linspace(0, 1, 100)*R
    z = np.linspace(-0.5, 0.5, 100)*R
    
    # equation constants
    lambda_m = []
    Br = 0
    Bz = 0

    # find first three bessel roots
    a_m = (sp.special.jn_zeros(1, 3))/h_scale
    
    # check
    print 'r = ', r
    print 'z = ', z
    print 'k = ', k
    print 'a_m = ', a_m
    
    # find first three lambda_m's
    for i in range(3):
      
      lambda_m.append((a_m[i]/R)**2)
    
    # check
    print 'lambda_m = ', lambda_m
    
    # make a mesh grid for contour plot
    x, y = np.meshgrid(r, z)
    
    # calculate first three summations of Br and Bz    
    for j in range(3):
      
      Br += B_r(x, y, k[j], lambda_m[j], h)
      Bz += B_z(x, y, k[j], lambda_m[j], h, B_0)
    
    ###SECOND###
    
    # substituding values for curve graphs
    z1 = 0
    z2 = 0.26*R
    
    for u in range(3):
      Bz0 = B_z(r, z1, k[u], lambda_m[u], h, B_0)*B_0
      Br0 = B_r(r, z2, k[u], lambda_m[u], h)*B_0
    
    ###PLOTTING###
    # plots 
    plt.figure(1, figsize=(8,4))
    plt.suptitle('Magnetic Field Plots', fontsize=12, fontweight='bold') # Title for whole figure
    plt.figtext(0.01,0.97,'Created by: ' + name, size=5) # Add created by to top left corner
    plt.figtext(0.01,0.95, 'Todays Date: '  + date, size=5) # Add date to top left corner
    plt.figtext(0.01,0.93,'Time:  ' + clock, size=5) # Add clock time to top left corner
    
    plt.subplot(1,2,1) 
    cp1 = plt.contour(x, y, Br)
    plt.title('\n\n$B_r$', fontsize=8)
    plt.xlabel('$r$', fontsize=8)
    plt.ylabel('$z$', fontsize=8)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.clabel(cp1, inline = True, fontsize = 3)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize = 5) # change number size on colorbar
    
    plt.subplot(1,2,2)
    cp2 = plt.contour(x, y, Bz)
    plt.title('\n\n$B_z$', fontsize=8)
    plt.xlabel('$r$', fontsize=8)
    plt.ylabel('$z$', fontsize=8)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.clabel(cp2, inline = True, fontsize = 3)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize = 5) # change number size on colorbar
    
    plt.tight_layout()
    plt.savefig('mag_field_plots_1.png', dpi = 800)
    plt.show()
    
    plt.figure(2, figsize=(8,4))
    plt.suptitle('Magnetic Field Plots', fontsize=12, fontweight='bold') # Title for whole figure
    plt.figtext(0.01,0.97,'Created by: ' + name, size=5) # Add created by to top left corner
    plt.figtext(0.01,0.95, 'Todays Date: '  + date, size=5) # Add date to top left corner
    plt.figtext(0.01,0.93,'Time:  ' + clock, size=5) # Add clock time to top left corner
    
    plt.subplot(1,2,1)
    cp2 = plt.contour(x, y, np.arctan(Bz/Br))
    plt.title('\n\n$title$', fontsize=8)
    plt.xlabel('$r$', fontsize=8)
    plt.ylabel('$z$', fontsize=8)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.clabel(cp2, inline = True, fontsize = 3)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize = 5) # change number size on colorbar
    
    plt.subplot(1,2,2)
    cp2 = plt.contour(x, y, np.sqrt((Bz**2)+(Br**2)))
    plt.title('\n\n$title$', fontsize=8)
    plt.xlabel('$r$', fontsize=8)
    plt.ylabel('$z$', fontsize=8)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.clabel(cp2, inline = True, fontsize = 3)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize = 5) # change number size on colorbar
    
    plt.tight_layout()
    plt.savefig('mag_field_plots_2.png', dpi = 800)
    plt.show()
    
    plt.figure(3, figsize=(8,4))
    plt.suptitle('Magnetic Field Plots', fontsize=12, fontweight='bold') # Title for whole figure
    plt.figtext(0.01,0.97,'Created by: ' + name, size=5) # Add created by to top left corner
    plt.figtext(0.01,0.95, 'Todays Date: '  + date, size=5) # Add date to top left corner
    plt.figtext(0.01,0.93,'Time:  ' + clock, size=5) # Add clock time to top left corner
    
    plt.subplot(1,2,1)
    plt.plot(r, Bz0)
    plt.title('\n\n$z-component$ $at$ $z=0$', fontsize=8)
    plt.xlabel('$r$', fontsize=8)
    plt.ylabel('$B_z$', fontsize=8)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(r, Br0)
    plt.title('\n\n$r-component$ $at$ $z=0.26R$', fontsize=8)
    plt.xlabel('$r$', fontsize=8)
    plt.ylabel('$B_r$', fontsize=8)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mag_field_plots_3.png', dpi = 800)
    plt.show()
    
    plt.figure(4, figsize=(8,8))
    plt.figtext(0.01,0.97,'Created by: ' + name, size=5) # Add created by to top left corner
    plt.figtext(0.01,0.95, 'Todays Date: '  + date, size=5) # Add date to top left corner
    plt.figtext(0.01,0.93,'Time:  ' + clock, size=5) # Add clock time to top left corner
    
    plt.quiver(x, y, Bz, Br, headlength=7)
    plt.title('\n\n$Model$ $of$ $Magnetic$ $Field$ $Lines$', fontsize=16)
    plt.xlabel('$r$', fontsize=14)
    plt.ylabel('$z$', fontsize=14)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend()
    
    # fit, save and show plot
    plt.tight_layout()
    plt.savefig('mag_field_plots_4.png', dpi = 800)
    plt.show()
    
def B_r(r, z, k_m, lambda_m, h):
    
    """
    This function 
    """
    
    # compontents
    a = k_m
    b = np.sqrt(lambda_m)
    c = h
    
    return a*b*sp.special.j1(b*r)*((sp.special.erfc((b*c)/2 - (z/c))*np.exp((-b*z))) - (sp.special.erfc((b*c)/2 + (z/c))*np.exp((b*z))))
    
def B_z(r, z, k_m, lambda_m, h, B_0):
    
    """   
    This function 
    """
    
    # compontents
    a = k_m
    b = np.sqrt(lambda_m)
    c = h
    
    return a*b*sp.special.j0(b*r)*((sp.special.erfc((b*c)/2 + (z/c))*np.exp((b*z))) + (sp.special.erfc((b*c)/2 - (z/c))*np.exp((-b*z))))+B_0
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    