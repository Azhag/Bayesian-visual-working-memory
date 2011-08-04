#!/usr/bin/env python
# encoding: utf-8
"""
utils.py

Created by Loic Matthey on 2011-06-16.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import pylab as plt
import numpy as np
import matplotlib.ticker as plttic

__maxexp__ = np.finfo('float').maxexp

def cross(*args):
    ans = [[]]
    for arg in args:
        if isinstance(arg[0], list) or isinstance(arg[0], tuple):
            for a in arg:
                ans = [x+[y] for x in ans for y in a]
        else:
            ans = [x+[y] for x in ans for y in arg]
    return ans


def plot_std_area(x, y, std, ax_handle=None):
    if ax_handle is None:
        f = plt.figure()
        ax_handle = f.add_subplot(111)
    
    ax = ax_handle.plot(x, y)
    current_color = ax[-1].get_c()
    
    ax_handle.fill_between(x, y-std, y+std, facecolor=current_color, alpha=0.4,
                        label='1 sigma range')
    
    return ax_handle

def array2string(array):
    # return np.array2string(array, suppress_small=True)
    if array.ndim == 2:
        return '  |  '.join([' '.join(str(k) for k in item) for item in array])
    elif array.ndim == 3:
        return '  |  '.join([', '.join([' '.join([str(it) for it in obj]) for obj in item]) for item in array])

def plot_square_grid(x, y, nb_to_plot=-1):
    '''
        Construct a square grid of plots
        
        Uses the first dimension as number of subplots.
    '''
    if nb_to_plot < 0:
        nb_to_plot = y.shape[0]
    
    nb_plots_sqrt = np.sqrt(nb_to_plot).astype(np.int32)
    f, subaxes = plt.subplots(nb_plots_sqrt, nb_plots_sqrt)
    
    for i in np.arange(nb_plots_sqrt):
        for j in np.arange(nb_plots_sqrt):
            try:
                subaxes[i,j].plot(x[nb_plots_sqrt*i+j], y[nb_plots_sqrt*i+j])
                subaxes[i,j].xaxis.set_major_locator(plttic.NullLocator())
                subaxes[i,j].yaxis.set_major_locator(plttic.NullLocator())
            except IndexError:
                subaxes[i,j].set_visible(False)
    
    return (f, subaxes)    


def pcolor_square_grid(data, nb_to_plot=-1):
    '''
        Construct a square grid of pcolor
        
        Uses the first dimension as number of subplots.
    '''
    if nb_to_plot < 0:
        nb_to_plot = data.shape[0]
    
    nb_plots_sqrt = np.ceil(np.sqrt(nb_to_plot)).astype(int)
    f, subaxes = plt.subplots(nb_plots_sqrt, nb_plots_sqrt)
    
    for i in np.arange(nb_plots_sqrt):
        for j in np.arange(nb_plots_sqrt):
            try:
                subaxes[i,j].pcolor(data[nb_plots_sqrt*i+j])
                subaxes[i,j].xaxis.set_major_locator(plttic.NullLocator())
                subaxes[i,j].yaxis.set_major_locator(plttic.NullLocator())
            except IndexError:
                subaxes[i,j].set_visible(False)
                
    return (f, subaxes)
    



if __name__ == '__main__':
    pass