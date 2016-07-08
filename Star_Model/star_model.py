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

# -*- coding: utf-8 -*-

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
#import mplot3d.axes3d as plt3
from mpl_toolkits.mplot3d import Axes3D
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
## Created on: July 5th, 2016
##
## Application Name: star_model.py
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

def magnetic_field(v_scale = 1, h_scale = 1, R = 3.8918, k1 = 0.9549, k2 = 0.4608, k3 = 0.6320, h = 0.3257, B_0 = 3.3118):
    
    """
    This function plots various magnetic field visualization graphs.
    """
    # select which plot to show
    # 1 for PLOT 1, 2 for PLOT 2, etc.
    # all for all plots
    # info for plot info
    print 'Which plot would you like to display?\n'
    for i in range(1,9): # number of plots total
      print 'For plot ', i, ', please type ', i
    print 'For all plots, please type all'
    print 'For info on all the plots, please type info'
    print_which = str(raw_input("Plot #: "))
    
    # model parameters
    k = [] # hold calculated k components
    k.append(k1) # k_1, k[0]
    k.append(k2) # k_2, k[1]
    k.append(k3) # k_3, k[2] 
    lambda_m = []
    lambda_m = lambda_m_calc(h_scale, R)

    ############################################################## 
    ### PLOT 1 ###################################################
    ##############################################################      
    
    if print_which == '1' or print_which == 'all':
      ############################################################  
      ## These two contour plots show the representations of 
      ## Br and Bz in terms of r and z components. 
      ############################################################
    
      ### PLOT 
      r = np.linspace(-1, 1, 100)*R
      z = np.linspace(-0.5, 0.5, 100)*R
      x, y = mesh_grid(r,z,R)
      Br, Bz = calc_Br_Bz(x,y,k,lambda_m,h,B_0)
      subtitle='$k_1 = {0}, k_2 = {1}, k_3 = {2}, R = {3}, h = {4}, B_0 = {5}$'.format(k1,k2,k3,R,h,B_0)
      figure_info(1, title='Magnetic Field Plot 1', subtitle=subtitle)
      plt.subplot(1,2,1)
      plot_contour(x,y,Br,title='\n\n$B_r$',xlab='$r$',ylab='$z$')
      plt.subplot(1,2,2)
      plot_contour(x,y,Bz,title='\n\n$B_z$',xlab='$r$',ylab='$z$')
      plt.tight_layout()
      plt.savefig('mag_field_plots_1.png', dpi = 800)
      plt.show()

    ############################################################## 
    ### PLOT 2 ###################################################
    ##############################################################      
    
    if print_which == '2' or print_which == 'all':
      ############################################################  
      ## These two contour plots show the representations of 
      ## arctan(Bz/Br) and sqrt(Br^2+Bz^2) in terms of r and 
      ## z components. 
      ############################################################
      
      ### PLOT
      r = np.linspace(-1, 1, 100)*R
      z = np.linspace(-0.5, 0.5, 100)*R
      x, y = mesh_grid(r,z,R)
      Br, Bz = calc_Br_Bz(x,y,k,lambda_m,h,B_0)
      subtitle='$k_1 = {0}, k_2 = {1}, k_3 = {2}, R = {3}, h = {4}, B_0 = {5}$'.format(k1,k2,k3,R,h,B_0)
      figure_info(2, title='Magnetic Field Plot 2',subtitle=subtitle)
      plt.subplot(1,2,1)
      plot_contour(x,y,np.arctan(Bz/Br),title='\n\n$tan^{-1}(B_z/B_r)$',xlab='$r$',ylab='$z$')
      plt.subplot(1,2,2)
      plot_contour(x,y,np.sqrt((Bz**2)+(Br**2)),title='\n\n$\sqrt{Bz^{2}+Br^{2}}$',xlab='$r$',ylab='$z$')
      plt.tight_layout()
      plt.savefig('mag_field_plots_2.png', dpi = 800)
      plt.show()

    ############################################################## 
    ### PLOT 3 ###################################################
    ##############################################################      
    
    if print_which == '3' or print_which == 'all':
      ############################################################  
      ## These line plots show the features of Br and Bz by using
      ## modified z components to get a 2D line graph. 
      ############################################################
      
      ### PLOT 
      r = np.linspace(0, 1, 100)*R
      Br0, Bz0 = calc_Br_Bz_mod(k,lambda_m,h,B_0,R,r)
      subtitle='$k_1 = {0}, k_2 = {1}, k_3 = {2}, R = {3}, h = {4}, B_0 = {5}$'.format(k1,k2,k3,R,h,B_0)
      figure_info(3, title='Magnetic Field Plot 3',subtitle=subtitle)
      plt.subplot(1,2,1)
      plot(r,Bz0,title='\n\n$z-component$ $at$ $z=0$',xlab = '$r$',ylab='$B_z$')
      plt.subplot(1,2,2)
      plot(r,Br0,title='\n\n$r-component$ $at$ $z=0.26R$',xlab = '$r$',ylab='$B_r$')
      plt.tight_layout()
      plt.savefig('mag_field_plots_3.png', dpi = 800)
      plt.show()
    
    ############################################################## 
    ### PLOT 4 ###################################################
    ##############################################################   
    
    if print_which == '4' or print_which == 'all':
      ############################################################  
      ## This plots a representation of the magnetic field based
      ## on the r, z, Br, and Bz components so we can look at the
      ## actual lines creatd by the formulas.
      ############################################################
      
      ### PLOT 
      r = np.linspace(-1, 1, 100)*R
      z = np.linspace(-0.5, 0.5, 100)*R
      x, y = mesh_grid(r,z,R)
      Br, Bz = calc_Br_Bz(x,y,k,lambda_m,h,B_0)
      subtitle='$k_1 = {0}, k_2 = {1}, k_3 = {2}, R = {3}, h = {4}, B_0 = {5}$'.format(k1,k2,k3,R,h,B_0)
      figure_info(4, title='Magnetic Field Plot 4',size = (8,8),subtitle=subtitle,subxpos=0.5, subypos=0.945)
      plot_streamplot(r,z,Br,Bz,title='\n\n$Model$ $of$ $Magnetic$ $Field$ $Lines$',xlab = '$r$',ylab='$z$')
      plt.tight_layout()
      plt.savefig('mag_field_plots_4.png', dpi = 800)
      plt.show()

    ##############################################################
    ### PLOT 5 ###################################################
    ##############################################################   
    
    if print_which == '5' or print_which == 'all':
      ############################################################  
      ## This is a plot of stokes parameters Q and U found from
      ## the Bx, By and Bz vector components. 
      ############################################################  
      ## WARNING
      ## If you change the number of points in r and z, you must
      ## change the corresponding points in the mgrid for yy and
      ## zz. As well as the values in the for loop for y and z 
      ## when you calculate Q and U and the array size for 
      ## Sq and Su.
      ############################################################  
      
      ### PLOT      
      r = np.linspace(-1, 1, 100)*R
      z = np.linspace(-0.5, 0.5, 100)*R
      x, y = np.meshgrid(r,z)
      xx, yy, zz = np.mgrid[-0.5:0.5:10j, -0.5:0.5:100j, -0.5:0.5:100j]*R
      rr = np.sqrt(xx*xx+yy*yy)      
      Br, Bz = calc_Br_Bz(rr,zz,k,lambda_m,h,B_0)
      Bx, By= calc_BxBy(Br,xx,yy)
      Bx *= density(xx, yy, zz, R, h)
      By *= density(xx, yy, zz, R, h)
      Bz *= density(xx, yy, zz, R, h)
      Q, U = calc_Sq_Su(By, Bz)
      subtitle='$k_1 = {0}, k_2 = {1}, k_3 = {2}, R = {3}, h = {4}, B_0 = {5}$'.format(k1, k2, k3, R, h, B_0)
      figure_info(5, title='Stokes Parameters: Q and U',subtitle=subtitle)
      plt.subplot(1,2,1)
      plot_contour(x,y,Q,title='\n\n$Q$',xlab='$r$',ylab='$z$')
      plt.subplot(1,2,2)
      plot_contour(x,y,U,title='\n\n$U$',xlab='$r$',ylab='$z$')
      plt.tight_layout()
      plt.savefig('stokes_param.png', dpi = 800)
      plt.show()
    
    ############################################################## 
    ### PLOT 6 ###################################################
    ##############################################################  
    
    if print_which == '6' or print_which == 'all':
      ############################################################  
      ## This is the mayavi3d version of the magnetic field
      ## in plot 4. This opens a seperate window where you
      ## can see a 3D visualization and rotate it about
      ## any axis.
      ############################################################  
      ## import mayavi as maya 
      ## Br = sqrt(Bx*Bx+By*By)
      ## theta = arctan(y/x)
      ## Bx = Br*cos(theta)
      ## By = Br*sin(theta)
      ############################################################  
      ## mayavi.mlab.quiver3d(x, y, z, u, v, w, ...)
      ##
      ## If 6 arrays, (x, y, z, u, v, w) are passed, the 
      ## 3 first arrays give the position of the arrows, and 
      ## the 3 last the components. They can be of any shape.
      ############################################################ 
      
      ### PLOT      
      xx, yy, zz = np.mgrid[-0.5:0.5:10j, -0.5:0.5:10j, -0.5:0.5:100j]*R
      rr = np.sqrt(xx*xx+yy*yy)
      Br, Bz = calc_Br_Bz(rr,zz,k,lambda_m,h,B_0)
      Bx, By= calc_BxBy(Br,xx,yy)
      Bx *= density(xx, yy, zz, R, h)
      By *= density(xx, yy, zz, R, h)
      Bz *= density(xx, yy, zz, R, h)
      ml.figure(fgcolor=(0, 0, 0),bgcolor=(0,0,0))
      ml.quiver3d(xx,yy,zz,Bx,By,Bz, colormap='spectral', mode='2ddash')
      ml.savefig('mag_field_mayavi.png')     
      ml.show()

    ############################################################## 
    ### PLOT 7 ###################################################
    ##############################################################      
    
    if print_which == '7' or print_which == 'all':
      ############################################################  
      ## This graph is the mplot3d version of the mayavi3d graph.
      ## You can set the rotation of the xy and z axis by
      ## entering an integer (in degrees) between -360 and 360
      ## for either axis.
      ############################################################      
      
      print '\n'      
      print 'Please enter the rotation parameters for plot 7.'
      print 'You can rotate this graph in the xy direction and \nthe z direction.'
      print 'The xy direction is set at 0 degrees to be looking \nat the x-axis.'
      print 'The z direction is set at 0 degrees to be vertically \noriented.'
      print 'Please enter an integer from -360 to 360'
      try:
        rotate_xy = int(raw_input("Rotation amount in xy direction (degrees): ")) 
      except:
        ###ERROR###
        print 'Error! Try again.'
        print 'Please enter an integer (in degrees) from -360 to 360'
      try:
        rotate_z = int(raw_input("Rotation amount in z direction (degrees): ")) 
      except:
        ###ERROR###
        print 'Error! Try again.'
        print 'Please enter an integer (in degrees) from -360 to 360'
      
      ### PLOT
      xx, yy, zz = np.mgrid[-0.5:0.5:10j, -0.5:0.5:10j, -0.5:0.5:100j]*R
      rr = np.sqrt(xx*xx+yy*yy)
      Br, Bz = calc_Br_Bz(rr,zz,k,lambda_m,h,B_0)
      Bx, By= calc_BxBy(Br,xx,yy)
      Bx *= density2(xx, yy, zz)
      By *= density2(xx, yy, zz)
      Bz *= density2(xx, yy, zz)
      subtitle='$k_1 = {0}, k_2 = {1}, k_3 = {2}, R = {3}, h = {4}, B_0 = {5}$'.format(k1,k2,k3,R,h,B_0)
      fig = figure_info(7, title='3D Magnetic Field Plot (6) using mplot3d',size = (8,8),subtitle=subtitle,subxpos=0.5, subypos=0.945)
      plot_3d_quiver(fig,xx,yy,zz,Bx,By,Bz,rotate_xy,rotate_z)      
      plt.tight_layout()
      plt.savefig('3D_mag_field_plots', dpi = 800)
      plt.show()

    ############################################################## 
    ### PLOT 8 ###################################################
    ############################################################## 

    if print_which == '8' or print_which == 'all':
      ############################################################  
      ## This graph is the mplot3d version of the mayavi3d graph.
      ## You can set the rotation of the xy and z axis by
      ## entering an integer (in degrees) between -360 and 360
      ## for either axis.
      ############################################################

      print '\n'      
      print 'Please enter the rotation parameters for plot 8.'
      print 'You can rotate this graph in the xy direction and \nthe z direction.'
      print 'The xy direction is set at 0 degrees to be looking \nat the x-axis.'
      print 'The z direction is set at 0 degrees to be vertically \noriented.'
      print 'Please enter an integer from -360 to 360'
      try:
        rotate_xy = int(raw_input("Rotation amount in xy direction (degrees): ")) 
      except:
        ###ERROR###
        print 'Error! Try again.'
        print 'Please enter an integer (in degrees) from -360 to 360'
      try:
        rotate_z = int(raw_input("Rotation amount in z direction (degrees): ")) 
      except:
        ###ERROR###
        print 'Error! Try again.'
        print 'Please enter an integer (in degrees) from -360 to 360'
      
      ### PLOT
      r = np.linspace(-1, 1, 100)*R
      z = np.linspace(-0.5, 0.5, 100)*R
      x, y = np.meshgrid(r,z)
      xx, yy, zz = np.mgrid[-0.5:0.5:10j, -0.5:0.5:100j, -0.5:0.5:100j]*R
      rr = np.sqrt(xx*xx+yy*yy)      
      Br, Bz = calc_Br_Bz(rr,zz,k,lambda_m,h,B_0)
      Bx, By= calc_BxBy(Br,xx,yy)
      Bx *= density(xx, yy, zz, R, h)
      By *= density(xx, yy, zz, R, h)
      Bz *= density(xx, yy, zz, R, h)
      Q, U = calc_Sq_Su(By, Bz)
      subtitle='$k_1 = {0}, k_2 = {1}, k_3 = {2}, R = {3}, h = {4}, B_0 = {5}$'.format(k1, k2, k3, R, h, B_0)
      fig = figure_info(8, title='Stokes Parameters: Q and U',subtitle=subtitle)
      #plt.subplot(1,2,1)
      plot_3d_contour(fig,x,y,Q,rotate_xy,rotate_z)
      #plt.subplot(1,2,2)
      #plot_3d_contour(fig,x,y,U,rotate_xy,rotate_z)
      plt.tight_layout()
      plt.savefig('3D_stokes_param', dpi = 800)
      plt.show()

    ############################################################## 
    ### INFO #####################################################
    ##############################################################      
    
    if print_which == 'info':
      ###DISPLAY GRAPH INFO###
      print graph_info()
    
    ############################################################## 
    ### ERROR ####################################################
    ##############################################################      
    
    if print_which not in ('1','2','3','4','5','6','7','8','all','info'):
      ###ERROR###
      print 'Error!'
      print 'Input', str(print_which), 'not recognized.'
      print 'Please input the graph number, all or info.'

def graph_info():
    
    """
    This function displays info for what each plot is.
    """
    
def plot_3d_contour(fig,x,y,z,rotate_xy,rotate_z):   
  
    """
    This function 
    """

    ax = fig.gca(projection='3d')
    ax.contour(x,y,z)
    ax.view_init(elev=90+rotate_z,azim=90+rotate_xy)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('y', fontsize=8)
    ax.set_zlabel('z', fontsize=8)
    
def plot_3d_quiver(fig,x,y,z,X,Y,Z,rotate_xy,rotate_z):   
  
    """
    This function 
    """
    
    ax = fig.gca(projection='3d')
    ax.quiver(x,y,z,X,Y,Z,length=0.05)
    ax.view_init(elev=0+rotate_z,azim=90+rotate_xy)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('y', fontsize=8)
    ax.set_zlabel('z', fontsize=8)
      
def calc_Sq_Su(By, Bz):

    """
    This function calculates the Q and U for Stokes parameters.
    """
    
    Sq = np.empty([100,100])
    Su = np.empty([100,100])
    
    for k in range(100):
      
      for j in range(100):
        
        Sq[j][k] = 0
        Su[j][k] = 0
        
        for i in range(10):
          
          Byy = By[i][j][k]
          Bzz = Bz[i][j][k]
          #print By[i][j][k]
          
          theta = np.arctan2(Bzz,Byy)
          theta_p = (np.pi/2) + theta
          
          Q = np.cos(2*theta_p)
          U = np.sin(2*theta_p)
          
          
          Sq[j][k] += Q
          Su[j][k] += U
          
    return Sq, Su
    
def calc_BxBy(Br,xx,yy):
    
    """
    This function calulates Bx and By for our 3D graph.
    """
    
    theta = np.arctan2(yy,xx) # use arctan2 for choosing the quadrant correctly

    Bx = Br*np.cos(theta)
    By = Br*np.sin(theta) 
    
    return Bx, By
    
def plot_streamplot(r,z,Br,Bz,title='',xlab = '',ylab=''):
    
    """
    This function creates a streamplot with four parameters (r,z,Br,Bz).
    """
    
    plt.streamplot(r, z, Br, Bz)
    plt.title(title, fontsize=16)
    plt.xlabel(xlab, fontsize=14)
    plt.ylabel(ylab, fontsize=14)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend()
    
def lambda_m_calc(h_scale,R):
    
    """
    This function calculates lambda_m for the Br and Bz functions.
    """
    
    lambda_m = []
    
    # find first three bessel roots
    a_m = (sp.special.jn_zeros(1, 3))/h_scale
    
    # find first three lambda_m's
    for i in range(3):
      
      lambda_m.append((a_m[i]/R)**2)
      
    return lambda_m
      
def mesh_grid(r,z,R):
  
    """
    This function creates a mesh grid for two parameters r and z. 
    """
    
    x, y = np.meshgrid(r, z)
    
    return x, y
    
def calc_Br_Bz(x,y,k,lambda_m,h,B_0):
    
    """
    This function calculates the values for Br and Bz given x and y values.
    """
    
    Br = 0
    Bz = 0
    
    # calculate first three summations of Br and Bz    
    for j in range(3):
      
      Br += B_r_calc(x, y, k[j], lambda_m[j], h)
      Bz += B_z_calc(x, y, k[j], lambda_m[j], h, B_0)
      
    return Br, Bz
    
def calc_Br_Bz_mod(k,lambda_m,h,B_0,R,r):
    
    """
    This function calculate Br and Bz with the required modified z values 
    of the model.
    """
    
    # substituting values for curve graphs
    z1 = 0
    z2 = 0.26*R
    Bz0 = 0
    Br0 = 0
    
    # calculate first three summations of Br and Bz with modified z's
    # for line graph
    for u in range(3):
      Bz0 += B_z_calc(r, z1, k[u], lambda_m[u], h, B_0)/B_0
      Br0 += B_r_calc(r, z2, k[u], lambda_m[u], h)/B_0
      
    return Br0, Bz0
      
def figure_info(fignum,title='',size=(8,4), subtitle='', subxpos=0.5, subypos=0.92):
  
    """
    This function is to set the info for a given figure. Parameters include
    fignum, title, and size.
    """
    
    fig = plt.figure(fignum, size)
    plt.suptitle(title, fontsize=12, fontweight='bold') # Title for whole figure   
    plt.figtext(subxpos,subypos,subtitle, size=6,horizontalalignment='center',verticalalignment='center')    
    plt.figtext(0.01,0.97,'Created by: ' + name, size=5) # Add created by to top left corner
    plt.figtext(0.01,0.95, 'Todays Date: '  + date, size=5) # Add date to top left corner
    plt.figtext(0.01,0.93,'Time:  ' + clock, size=5) # Add clock time to top left corner
    
    return fig

def plot_contour(x,y,z,title='',xlab = '',ylab=''):
    
    """
    This function is for making a contour plot with three parameters (x,y,z).
    """
    
    cp = plt.contour(x, y, z)
    plt.title(title, fontsize=8)
    plt.xlabel(xlab, fontsize=8)
    plt.ylabel(ylab, fontsize=8)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.clabel(cp, inline = True, fontsize = 3)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize = 5) # change number size on colorbar
    
def plot(x,y,title='',xlab = '',ylab=''):
    
    """
    This function is for making a simple plot with two parameters (x,y).
    """ 
    
    plt.plot(x, y)
    plt.title(title, fontsize=8)
    plt.xlabel(xlab, fontsize=8)
    plt.ylabel(ylab, fontsize=8)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.legend()
    
def linspace(a,b,c):
  
    """
    This function returns a linear space.
    
    r = (-1,1)
    z = (-0.5,0.5)
    """
    
    return np.linspace(a, b, c)
    
def B_r_calc(r,z,k_m,lambda_m,h):
    
    """
    This function calculates and returns an expression for Br.
    """
    
    # compontents
    a = k_m
    b = np.sqrt(lambda_m)
    c = h
    
    return a*b*sp.special.j1(b*r)*((sp.special.erfc((b*c)/2 - (z/c))*np.exp((-b*z))) - (sp.special.erfc((b*c)/2 + (z/c))*np.exp((b*z))))
    
def B_z_calc(r,z,k_m,lambda_m,h,B_0):
    
    """   
    This function calculates and returns an expression for Bz.
    """
    
    # compontents
    a = k_m
    b = np.sqrt(lambda_m)
    c = h
    
    return a*b*sp.special.j0(b*r)*((sp.special.erfc((b*c)/2 + (z/c))*np.exp((b*z))) + (sp.special.erfc((b*c)/2 - (z/c))*np.exp((-b*z))))+B_0
    
def density(x, y, z, R, h):
    
    """   
    This function returns an expression for density.
    """
    
    # parameters
    row_0 = 10**7
    a = 1
    b = 1
    ### turnover at: a=68, b=68
    ### At 68, Q changes dramatically but U looks similar still
    
    return row_0*np.exp(-a*(x*x+y*y)/R*R)*np.exp(-b*(z*z)/h*h)
    
def density2(x, y, z):
    
    """   
    This function returns an expression for density.
    """
    
    # parameters
    row_0 = 10**7
    A = 0.3
    B = 0.3
    
    return row_0*np.exp(-(x*x+y*y)/A*A)*np.exp(-(z*z)/B*B)
        