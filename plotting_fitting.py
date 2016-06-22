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
import math as m
import matplotlib.pyplot as plt
#import pyqtgraph as pyqt
#import wxpython as wxp

#import pandas
import random
import time 
#import pdb

from scipy.optimize import curve_fit

## ###################################
##
## Author: Travis Thieme
## Created on: June 20th, 2016
##
## Application Name: Plotting and Fitting
## Programming Language: Python
## 
## Description: This program plots graphs of Sin, Cos, Gaussian, Bessel 
## and ERFC curves. It also can generate a random set of data and fit it 
## to one of these curves. 
## 
## ###################################

# Global Variables
name = 'Travis Thieme'
date  = time.strftime("%d/%m/%Y")

def project_info(project = ''):

    """
    This function prints the info for the project your working on. Info such as author,
    date, current time, and the project name
    """
    
    name = 'Travis Thieme'
    date = time.strftime("%A, %d/%m/%Y")
    clock = time.strftime("%H:%M:%S %Z")
    project_name = project

    print name
    print date
    print clock
    print project_name
    
def graph_sin():
    
    """
    This function graphs a simple sin curve from 0 to 2pi. 
    """ 
    # linearly spaced numbers from 0 to 2pi with 100 increments
    x = np.linspace(0,2*m.pi,100)
    y = np.sin(x) # sin function
    
    plt.clf() # clear plot
    plt.plot(x,y) # plot 
    plt.title('Sin Plot') # plot title
    plt.figtext(0.01,0.95,'Created by: ' + name) # Add created by to top left corner
    plt.figtext(0.72,0.95, 'Todays Date: '  + date) # Add date to top right corner
    plt.xlabel('x') # label x axis
    plt.ylabel('y') # label y axis
    plt.axis([0,2*m.pi,-1,1]) # x and y axis [x0,xf,yi,yf]
    plt.grid(True) # show grid lines
    plt.savefig('sin.png') # save figure to png file    
    plt.show() # show plot

def graph_cos():
    
    """
    This function graphs a simple sin curve from 0 to 2pi. 
    """ 
    
    # linearly spaced numbers from 0 to 2pi with 100 increments
    x = np.linspace(0,2*m.pi,100)
    y = np.cos(x) # cos function
    
    plt.clf() # clear plot
    plt.plot(x,y) # plot 
    plt.title('Cos Plot') # plot title
    plt.figtext(0.01,0.95,'Created by: ' + name) # Add created by to top left corner
    plt.figtext(0.72,0.95, 'Todays Date: '  + date) # Add date to top right corner
    plt.xlabel('x') # label x axis
    plt.ylabel('y') # label y axis
    plt.axis([0,2*m.pi,-1,1]) # x and y axis [x0,xf,yi,yf]
    plt.grid(True) # show grid lines
    plt.savefig('cos.png') # save figure to png file   
    plt.show() # show plot 
    
def graph_gaussian():
    
    """
    This function graphs a gaussian distribution curve from a data set. 
    """ 
    
    # set values for gaussian distribution
    A = 10
    x_0 = 0
    sigma = 0.5
    
    # linearly spaced numbers from -10 to 10 with 100 increments
    x = np.linspace(-10,10,100)
    y = gaussian(x, A, x_0, sigma) 
    noise = random_noise() # add noise to function
    y_2 = y + noise # new function with noise
    
    print '\n'    
    print 'Noisy Data List: '
    print noise # check
    
    params, extras = curve_fit(gaussian, x, y_2) # fit curve to function
    y_3 = gaussian(x, params[0], params[1], params[2]) # found parameters
    
    # print results
    print '\n'    
    print 'Actual Gaussian Distribution values:'
    print 'A = ', A
    print 'x_0 = ', x_0
    print 'sigma = ', sigma
    print '\n'
    print 'Fitted Gaussian Distribution values:'
    print 'A = ', params[0]
    print 'x_0 = ', params[1]
    print 'sigma = ', params[2]

    plt.clf() # clear plot
    plt.plot(x,y, 'b--', label = 'origional') # plot origional
    plt.plot(x,y_2, 'ro', label = 'noise') # plot noise dots
    plt.plot(x, y_3, 'g', label = 'fitted') # plot fit
    plt.title('Gaussian Plot') # plot title
    plt.figtext(0.01,0.95,'Created by: ' + name) # Add created by to top left corner
    plt.figtext(0.72,0.95, 'Todays Date: '  + date) # Add date to top right corner
    plt.xlabel('x') # label x axis
    plt.ylabel('y') # label y axis
    plt.legend() # show legend
    plt.grid(True) # show grid lines
    plt.savefig('gaussian.png') # save figure to png file
    plt.show() # show plot

def gaussian(x, A, x_0, sigma):  

    """
    This function generates a gaussian distribution 
    """
    
    return A*np.exp(-(x-x_0)**2/(2.*(sigma)**2))
       
def random_noise():
    
    """
    This function generates a list of random noise. 
    """ 
    noise = [] # empty data list
    
    # fill data list with random numbers
    for i in range (100):
      noise.append(random.uniform(-2.0,2.0))
      
    return noise # return data

def graph_bessel_1():
    
    """
    This function graphs the J_1 bessel function. 
    """ 
    # set values for gaussian distribution
    A = 5
    b = 1
    
    # linearly spaced numbers from -10 to 10 with 100 increments
    x = np.linspace(-20,20,100)
    y = bessel_1(x, A, b)
    noise = random_noise() # add noise to function
    y_2 = y + noise # new function with noise
    
    params, extras = curve_fit(bessel_1, x, y_2) # fit curve to function
    print params
    y_3 = bessel_1(x, params[0], params[1]) # found parameters
    
    print '\n'    
    print 'Noisy Data List: '
    print noise # check
    # print results
    print '\n'    
    print 'Actual Bessel Distribution values:'
    print 'A = ', A
    print 'b = ', b
    print '\n'
    print 'Fitted Gaussian Distribution values:'
    print 'A = ', params[0]
    print 'b = ', params[1]
    
    plt.clf() # clear plot
    plt.title('Bessel Plot') # plot title
    plt.figtext(0.01,0.95,'Created by: ' + name) # Add created by to top left corner
    plt.figtext(0.72,0.95, 'Todays Date: '  + date) # Add date to top right corner
    plt.plot(x,y, 'b--', label = 'origional') # plot origional
    plt.plot(x,y_2, 'ro', label = 'noise') # plot noise dots
    plt.plot(x, y_3, 'g', label = 'fitted') # plot fit
    plt.xlabel('x') # label x axis
    plt.ylabel('y') # label y axis
    plt.legend() # show legend
    plt.grid(True) # show grid lines
    plt.savefig('bessel_j1.png') # save figure to png file
    plt.show() # show plot
    
def bessel_1(x, A, b):  

    """
    This function calculates the J1 bessel function
    """
    
    return A*sp.special.j1(b*x)
    
def graph_bessel_0():
    
    """
    This function graphs the J_0 bessel function. 
    """ 
    # set values for gaussian distribution
    A = 5
    b = 1
    
    # linearly spaced numbers from -10 to 10 with 100 increments
    x = np.linspace(-20,20,100)
    y = bessel_0(x, A, b)
    noise = random_noise() # add noise to function
    y_2 = y + noise # new function with noise
    
    params, extras = curve_fit(bessel_0, x, y_2) # fit curve to function
    print params
    y_3 = bessel_0(x, params[0], params[1]) # found parameters
    
    print '\n'    
    print 'Noisy Data List: '
    print noise # check
    # print results
    print '\n'    
    print 'Actual Bessel Distribution values:'
    print 'A = ', A
    print 'b = ', b
    print '\n'
    print 'Fitted Gaussian Distribution values:'
    print 'A = ', params[0]
    print 'b = ', params[1]
    
    plt.clf() # clear plot
    plt.title('Bessel Plot') # plot title
    plt.figtext(0.01,0.95,'Created by: ' + name) # Add created by to top left corner
    plt.figtext(0.72,0.95, 'Todays Date: '  + date) # Add date to top right corner
    plt.plot(x,y, 'b--', label = 'origional') # plot origional
    plt.plot(x,y_2, 'ro', label = 'noise') # plot noise dots
    plt.plot(x, y_3, 'g', label = 'fitted') # plot fit
    plt.xlabel('x') # label x axis
    plt.ylabel('y') # label y axis
    plt.legend() # show legend
    plt.grid(True) # show grid lines
    plt.savefig('bessel_j0.png') # save figure to png file
    plt.show() # show plot
    
def bessel_0(x, A, b):  

    """
    This function calculates the J0 bessel function
    """
    
    return A*sp.special.j0(b*x)
    
def graph_ERFC():
    
    """
    This function graphs the error function. 
    """ 
    
    # set values for gaussian distribution
    C = 5
    d = 2
    e = 3
    
    # linearly spaced numbers from -10 to 10 with 100 increments
    x = np.linspace(-20,20,100)
    y = ERFC(x, C, d, e)
    noise = random_noise() # add noise to function
    y_2 = y + noise # new function with noise
    
    params, extras = curve_fit(ERFC, x, y_2) # fit curve to function
    print params
    y_3 = ERFC(x, params[0], params[1], params[2]) # found parameters
    
    plt.clf() # clear plot
    plt.title('ERFC Plot') # plot title
    plt.figtext(0.01,0.95,'Created by: ' + name) # Add created by to top left corner
    plt.figtext(0.72,0.95, 'Todays Date: '  + date) # Add date to top right corner
    plt.plot(x,y, 'b--', label = 'origional') # plot origional
    plt.plot(x,y_2, 'ro', label = 'noise') # plot noise dots
    plt.plot(x, y_3, 'g', label = 'fitted') # plot fit
    plt.xlabel('x') # label x axis
    plt.ylabel('y') # label y axis
    plt.legend(loc=0) # show legend
    plt.grid(True) # show grid lines
    plt.savefig('erfc.png') # save figure to png file
    plt.show() # show plot
    
def ERFC(x, C, d, e):  

    """
    This function calculates the error function
    """
    
    return C*sp.special.erfc(d-(x/e))
    
def graph_bessel_ERFC():
  
    """
    This function graphs the bessel function * the error function. 
    """
    
    # set values for gaussian distribution
    A = 5
    b = 2
    c = 1
    d = 1
    
    # linearly spaced numbers from -10 to 10 with 100 increments
    x = np.linspace(-20,20,50)
    y = bessel_ERFC(x, A, b, c, d)
    noise = random_noise_BE() # add noise to function
    y_2 = y + noise # new function with noise
    
    params, extras = curve_fit(bessel_ERFC, x, y_2) # fit curve to function
    print params
    y_3 = bessel_ERFC(x, params[0], params[1], params[2], params[3]) # found parameters
    
    plt.clf() # clear plot
    plt.title('Bessel*ERFC Plot') # plot title
    plt.figtext(0.01,0.95,'Created by: ' + name) # Add created by to top left corner
    plt.figtext(0.72,0.95, 'Todays Date: '  + date) # Add date to top right corner
    plt.plot(x,y, 'b--', label = 'origional') # plot origional
    plt.plot(x,y_2, 'ro', label = 'noise') # plot noise dots
    plt.plot(x, y_3, 'g', label = 'fitted') # plot fit
    plt.xlabel('x') # label x axis
    plt.ylabel('y') # label y axis
    plt.legend() # show legend
    plt.grid(True) # show grid lines
    plt.savefig('bessel_erft.png') # save figure to png file
    plt.show() # show plot
    
def bessel_ERFC(x, A, b, c, d):
  
    return A*sp.special.j1(b*x)*sp.special.erfc(c-(x/d))
    
def random_noise_BE():
    
    """
    This function generates a list of random noise. 
    """ 
    noise = [] # empty data list
    
    # fill data list with random numbers
    for i in range (50):
      noise.append(random.uniform(-0.2,0.2))
      
    return noise # return data