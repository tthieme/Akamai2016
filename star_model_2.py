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
import mayavi.mlab as ml
#import pyqtgraph as pyqt
#import wx as wxp

#import pandas
#import random
import time 
#import pdb


## ###################################
##
## Author: Travis Thieme
## Created on: June 29th, 2016
##
## Application Name: 
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

def final():
  
    """
    This function 
    """
    
    

def magnetic_field(v_scale = 1, h_scale = 1, R = 3.8918, k1 = 0.9549, k2 = 0.4608, k3 = 0.6320, h = 0.3257, B_0 = 3.3118):
    
    """
    This function 
    """
    
    # model parameters
    k = [] # hold calculated k components
    k.append(k1) # k_1, k[0]
    k.append(k2) # k_2, k[1]
    k.append(k3) # k_3, k[2]
    
    lambda_m = []
    lambda_m = lambda_m_calc(h_scale, R)
    
    ###PLOT 1###
    x, y = mesh_grid(0,1,100,-0.5,0.5,100)
    Br, Bz = calc_Br_Bz(x,y,k,lambda_m,h,B_0)
    
      
def lambda_m_calc(h_scale, R):
    
    """
    This function 
    """
    
    lambda_m = []
    
    # find first three bessel roots
    a_m = (sp.special.jn_zeros(1, 3))/h_scale
    
    # find first three lambda_m's
    for i in range(3):
      
      lambda_m.append((a_m[i]/R)**2)
      
    return lambda_m
      
def mesh_grid(a,b,c,d,e,f):
  
    """
    This function 
    """
    r = np.linspace(a,b,100) 
    z = np.linspace(-0.5,0.5,100)
    
    x, y = np.meshgrid(r, z)
    
    return x, y
    
def calc_Br_Bz(x,y,k,lambda_m,h,B_0):
    
    """
    This function 
    """
    
    Br = 0
    Bz = 0
    
    # calculate first three summations of Br and Bz    
    for j in range(3):
      
      Br += B_r_calc(x, y, k[j], lambda_m[j], h)
      Bz += B_z_calc(x, y, k[j], lambda_m[j], h, B_0)
      
    return Br, Bz
      
def figure_info(fignum, title=''):
  
    """
    This function 
    """
    
    plt.figure(fignum, figsize=(8,4))
    plt.suptitle(title, fontsize=12, fontweight='bold') # Title for whole figure
    plt.figtext(0.01,0.97,'Created by: ' + name, size=5) # Add created by to top left corner
    plt.figtext(0.01,0.95, 'Todays Date: '  + date, size=5) # Add date to top left corner
    plt.figtext(0.01,0.93,'Time:  ' + clock, size=5) # Add clock time to top left corner
    

def plot_contour(x,y,z,title='',xlab = '',ylab=''):
  
    cp1 = plt.contour(x, y, z)
    plt.title(title, fontsize=8)
    plt.xlabel(xlab, fontsize=8)
    plt.ylabel(ylab, fontsize=8)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.clabel(cp1, inline = True, fontsize = 3)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize = 5) # change number size on colorbar
    
def linspace(a, b, c):
  
    """
    This function 
    
    r = (-1,1)
    z = (-0.5,0.5)
    """
    
    return np.linspace(a, b, c)
    
def B_r_calc(r, z, k_m, lambda_m, h):
    
    """
    This function eturns an expression for Br
    """
    
    # compontents
    a = k_m
    b = np.sqrt(lambda_m)
    c = h
    
    return a*b*sp.special.j1(b*r)*((sp.special.erfc((b*c)/2 - (z/c))*np.exp((-b*z))) - (sp.special.erfc((b*c)/2 + (z/c))*np.exp((b*z))))
    
def B_z_calc(r, z, k_m, lambda_m, h, B_0):
    
    """   
    This function returns an expression for Bz
    """
    
    # compontents
    a = k_m
    b = np.sqrt(lambda_m)
    c = h
    
    return a*b*sp.special.j0(b*r)*((sp.special.erfc((b*c)/2 + (z/c))*np.exp((b*z))) + (sp.special.erfc((b*c)/2 - (z/c))*np.exp((-b*z))))+B_0
    
    