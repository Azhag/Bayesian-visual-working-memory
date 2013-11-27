#!/usr/bin/env python
# encoding: utf-8

import matplotlib.pyplot as plt
import numpy as np

from launchers_memorycurves import *


def memory_curve_plot():
    
    plt.rcParams['font.size'] = 17

    loaded_data1 = np.load('/Users/loicmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/documents/FI_estimation_correctfits/131112_expvol/theo/data/theomemcurv_rcspace_sigmaspace_highprecision-launcher_do_memorycurve_theoretical-a35232ba-574f-436c-9e8e-082036785059.npy').item()
    
    plots_memorycurve_theoretical(loaded_data1, save_figures=False)

    # Keep figure 2, the best theoretical memory curve fit.
    plt.figure(2)
    plt.xticks(loaded_data1['T_space'], fontsize=13)
    plt.xlim([0.8, 5.2])
    plt.title('')

    # also treat fig 1, space of FI mem curves fit
    plt.figure(1)

    plt.title('')

    # fix ticks
    nb_ticks_x = 6
    x_ticks_i = np.linspace(0, loaded_data1['rcscale_space'].size-1, nb_ticks_x)
    x_ticks_labels = np.linspace(loaded_data1['rcscale_space'].min(), loaded_data1['rcscale_space'].max(), nb_ticks_x)
    plt.xticks(x_ticks_i, ["%.2f" % curr for curr in x_ticks_labels])
    plt.xlabel('kappa')

    nb_ticks_y = 6
    y_ticks_i = np.linspace(0, loaded_data1['sigma_space'].size-1, nb_ticks_y)
    y_ticks_labels = np.linspace(loaded_data1['sigma_space'].min(), loaded_data1['sigma_space'].max(), nb_ticks_y)
    plt.yticks(y_ticks_i, ["%.2f" % curr for curr in y_ticks_labels])
    plt.ylabel('sigma')

    
    loaded_data2 = np.load('/Users/loicmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/documents/FI_estimation_correctfits/131112_expvol/theo/data/PLOTS_theomemcurv_rcspace_sigmaspace-launcher_do_memorycurve_theoretical-6956d326-c2e8-4be8-8de1-3e554bac69d8.npy').item()

    
    plots_memorycurve_theoretical(loaded_data2, save_figures=False)

    # Keep figure 3
    plt.figure(3)
    plt.close(4)

    plt.title('')

    # fix ticks
    nb_ticks_x = 6
    x_ticks_i = np.linspace(0, loaded_data2['rcscale_space'].size-1, nb_ticks_x)
    x_ticks_labels = np.linspace(loaded_data2['rcscale_space'].min(), loaded_data2['rcscale_space'].max(), nb_ticks_x)
    plt.xticks(x_ticks_i, ["%.2f" % curr for curr in x_ticks_labels])
    plt.xlabel('kappa')

    nb_ticks_y = 6
    y_ticks_i = np.linspace(0, loaded_data2['sigma_space'].size-1, nb_ticks_y)
    y_ticks_labels = np.linspace(loaded_data2['sigma_space'].min(), loaded_data2['sigma_space'].max(), nb_ticks_y)
    plt.yticks(y_ticks_i, ["%.2f" % curr for curr in y_ticks_labels])
    plt.ylabel('sigma')

    plt.show()



if __name__ == '__main__':
    memory_curve_plot()

