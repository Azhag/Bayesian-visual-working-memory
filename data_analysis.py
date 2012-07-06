#!/usr/bin/env python
# encoding: utf-8
"""
data_analysis.py


Created by Loic Matthey on 2011-11-18.
Copyright (c) 2011 . All rights reserved.
"""

import os
import glob
import re
import cPickle
#from enthought.mayavi import mlab
from matplotlib.image import AxesImage
from string import Template
from optparse import OptionParser

import numpy as np
import scipy.special as scsp
from scipy.stats import vonmises as vm
import scipy.optimize as spopt
import scipy.interpolate as spint
import time
import sys
import os.path
import argparse
import pylab as plt

from datagenerator import *
from randomnetwork import *
from randomfactorialnetwork import *
from statisticsmeasurer import *
from slicesampler import *
from utils import *


########################


def combine_pbs_effectT():
    """
        Combine simulation outputs from PBS
    """
    
    # Get all output files
    # all_output_files = glob.glob('conj_effectT*.npy*')
    all_output_files = glob.glob('feat_effectT*.npy*')
    
    # Iterate over them, load the corresponding signal and compute its metric
    all_T = []
    all_precisions = {}

    for curr_file in all_output_files:
        
        # Do a nice regular expression to catch the parameters and remove the useless random unique string
        matched = re.search('^[a-zA-Z_]*-T([0-9]*)alpha[0-9.]*N[0-9.]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)size_receptive_field_number_neurons-([0-9a-z\-]*).npy', curr_file)
        
        
        # Get N
        curr_T = int(matched.groups()[0])
        all_T.append(curr_T)

        # Get nbrepeats and nbexperiments
        nb_repeats = int(matched.groups()[1])
        nb_experiments = int(matched.groups()[2])
        
        print str(curr_T)
        
        # Load the data
        curr_output = np.load(curr_file).item()
        
        # Reload some things
        param1_space = curr_output['param1_space']
        param2_space = curr_output['param2_space']
        args = curr_output['args']
        curr_precisions = curr_output['all_precisions']

        # Store it
        if curr_T in all_precisions:
            all_precisions[curr_T].append(curr_precisions)
        else:
            all_precisions[curr_T] = [curr_precisions]
    
    all_T = np.sort(np.unique(all_T))
    
    # Now have to put everything in a nice 4D array...
    results_array = np.zeros((all_T.size, param1_space.size, param2_space.size, nb_repeats*nb_experiments))
    for Ti in np.arange(all_T.size):
        for par1 in np.arange(param1_space.size):
            for par2 in np.arange(param2_space.size):
                for exp in np.arange(nb_experiments):
                    try:
                        results_array[Ti, par1, par2, nb_repeats*(exp):nb_repeats*(exp+1)] = all_precisions[all_T[Ti]][exp][par1][par2]
                    except IndexError:
                        pass

    
    mean_precisions = np.mean(1./results_array, 3)
    var_precisions = np.std(1./results_array, 3)

    # Some quick/dirty plots
    for t in np.arange(all_T.size):
        f = plt.figure()
        ax = f.add_subplot(111)
        im = ax.imshow(mean_precisions[t].T, origin='lower', aspect='auto')
        # im.set_extent((param1_space.min(), param1_space.max(), param2_space.min(), param2_space.max()))
        im.set_interpolation('nearest')
        # ax.xaxis.set_major_locator(plttic.NullLocator())
        # ax.yaxis.set_major_locator(plttic.NullLocator())
        plt.xticks(np.arange(param1_space.size), param1_space)
        plt.yticks(np.arange(param2_space.size), np.around(param2_space, 2))
        ax.set_xlabel('Number of neurons')
        ax.set_ylabel('Scale of receptive field')
        f.colorbar(im)
        ax.set_title('Precision for T=%d' % all_T[t])

    

def combine_plot_size_receptive_field_number_neurons():
    # Get all output files
    all_output_files = glob.glob('Data/conj_T2_2011_r3*.npy*')
    
    # Iterate over them, load the corresponding signal and compute its metric
    all_T = []
    all_precisions = {}

    for curr_file in all_output_files:
        
        # Do a nice regular expression to catch the parameters and remove the useless random unique string
        matched = re.search('^[a-zA-Z_\/]*_T([0-9]*)_[0-9]*_r([0-9.]*)_size_receptive_field_number_neurons-([0-9a-z\-]*).npy', curr_file)
        
        # Get N
        curr_T = int(matched.groups()[0])
        all_T.append(curr_T)

        # Get nbrepeats and nbexperiments
        nb_repeats = int(matched.groups()[1])
        nb_experiments = len(all_output_files)

        print str(curr_T)
        
        # Load the data
        curr_output = np.load(curr_file).item()
        
        # Reload some things
        param1_space = curr_output['param1_space']
        param2_space = curr_output['param2_space']
        args = curr_output['args']
        curr_precisions = curr_output['all_precisions']

        # Store it
        if curr_T in all_precisions:
            all_precisions[curr_T].append(curr_precisions)
        else:
            all_precisions[curr_T] = [curr_precisions]
    
    all_T = np.sort(np.unique(all_T))
    
    # Now have to put everything in a nice 4D array...
    results_array = np.zeros((all_T.size, param1_space.size, param2_space.size, nb_repeats*nb_experiments))
    for Ti in np.arange(all_T.size):
        for par1 in np.arange(param1_space.size):
            for par2 in np.arange(param2_space.size):
                for exp in np.arange(nb_experiments):
                    try:
                        results_array[Ti, par1, par2, nb_repeats*(exp):nb_repeats*(exp+1)] = all_precisions[all_T[Ti]][exp][par1][par2]
                    except IndexError:
                        pass

    
    mean_precisions = np.mean(1./results_array, 3)
    var_precisions = np.std(1./results_array, 3)

    # Some quick/dirty plots
    for t in np.arange(all_T.size):
        f = plt.figure()
        ax = f.add_subplot(111)
        im = ax.imshow(mean_precisions[t].T, origin='lower', aspect='auto')
        # im.set_extent((param1_space.min(), param1_space.max(), param2_space.min(), param2_space.max()))
        im.set_interpolation('nearest')
        # ax.xaxis.set_major_locator(plttic.NullLocator())
        # ax.yaxis.set_major_locator(plttic.NullLocator())
        plt.xticks(np.arange(param1_space.size), param1_space)
        plt.yticks(np.arange(param2_space.size), np.around(param2_space, 2))
        ax.set_xlabel('Number of neurons')
        ax.set_ylabel('Scale of receptive field')
        f.colorbar(im)
        ax.set_title('Precision for T=%d' % all_T[t])


def combine_mixed_two_scales(data_to_use = 0, should_plot=True):

    # T1
    # params = dict(label='T1', files='Data/mixed_ratio_2scales_search/T1_N100/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')
    if data_to_use == 0:
        # params = dict(label='T1_used_figures_collated_bis_2311_midnight', files='Data/mixed_ratio_2scales_search/T1_N300_K50/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*K[0-9]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')
        # params = dict(label='T1_inabstract', files='Data/mixed_ratio_2scales_search/T1_N300_K100/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*K[0-9]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')
        # params = dict(label='T1', files='Data/mixed_ratio_2scales_search/K300/T1/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*K[0-9]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')

        # New datagen, N300
        # params = dict(label='T1_newdata_n300', files='Data/mixed_ratio_2scales_search/new_datagen/N300/T1/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')

        # New datagen, N500
        params = dict(label='T1_newdata_n500', files='Data/mixed_ratio_2scales_search/new_datagen/N500/T1/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')

    # T2
    # params = dict(label='T2', files='Data/mixed_ratio_2scales_search/T2_N300_K100/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*K[0-9]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')
    if data_to_use == 1:
        # params = dict(label='T2_used_figures_collated_bis_2311_midnight', files='Data/mixed_ratio_2scales_search/T2_N300_K100/less_data/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*K[0-9]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')
        params = dict(label='T2_inabstract', files='Data/mixed_ratio_2scales_search/T2_N300_K100/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*K[0-9]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')
        # params = dict(label='T2', files='Data/mixed_ratio_2scales_search/K300/T2/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*K[0-9]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')

        # New datagen, N300
        # params = dict(label='T2_newdata_n300', files='Data/mixed_ratio_2scales_search/new_datagen/N300/T2/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')

        # New datagen, N500
        params = dict(label='T2_newdata_n500', files='Data/mixed_ratio_2scales_search/new_datagen/N500/T2/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')
    
    # T3
    if data_to_use == 2:
        # params = dict(label='T3_used_figures_collated_bis_2311_midnight', files='Data/mixed_ratio_2scales_search/T3/less_data/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*K[0-9]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')
        # params = dict(label='T3_inabstract', files='Data/mixed_ratio_2scales_search/T3/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*K[0-9]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')
        # params = dict(label='T3', files='Data/mixed_ratio_2scales_search/K300/T3/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*K[0-9]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')

        # New datagen, N300
        # params = dict(label='T3_newdata_n300', files='Data/mixed_ratio_2scales_search/new_datagen/N300/T3/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')

        # New datagen, N500
        params = dict(label='T3_newdata_n500', files='Data/mixed_ratio_2scales_search/new_datagen/N500/T3/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')
    
    # T4
    if data_to_use == 3:
        # params = dict(label='T4_inabstract', files='Data/mixed_ratio_2scales_search/T4/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*K[0-9]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')

        # New datagen, K300
        params = dict(label='T4_newdata_n300', files='Data/mixed_ratio_2scales_search/new_datagen/N300/T4/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')

        # New datagen, N500
        # params = dict(label='T4_newdata_n500', files='Data/mixed_ratio_2scales_search/new_datagen/N500/T4/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')
    

    # Get all output files
    all_output_files = glob.glob(params['files'])
    # all_output_files = glob.glob('Data/mixed_ratio_2scales_search/T2/ratioconj_2scales*.npy*')
    # all_output_files = glob.glob('Data/mixed_ratio_2scales_search/T2_N300_K50/cpy/ratioconj_2scales*.npy*')
    # all_output_files = glob.glob('Data/mixed_ratio_2scales_search/T2_N300_K100/ratioconj_2scales*.npy*')
    
    assert len(all_output_files) > 0, "Wrong regular expression"

    # Iterate over them, load the corresponding signal and compute its metric
    all_ratioconj = []
    all_precisions = {}

    for curr_file in all_output_files:
        
        # Do a nice regular expression to catch the parameters and remove the useless random unique string
        matched = re.search(params['regexp'], curr_file)
        # matched = re.search('^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy', curr_file)
        # matched = re.search('^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*K[0-9]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy', curr_file)
        
        
        # Get ratioconj
        curr_ratioconj = float(matched.groups()[0])
        all_ratioconj.append(curr_ratioconj)

        # Get nbrepeats and nbexperiments
        nb_repeats = int(matched.groups()[1])
        nb_experiments = int(len(all_output_files))

        print str(curr_ratioconj)
        
        # Load the data
        curr_output = np.load(curr_file).item()
        
        # Reload some things
        param1_space    = curr_output['param1_space']
        param2_space    = curr_output['param2_space']
        args            = curr_output['args']
        curr_precisions = curr_output['all_precisions']

        # Store it
        if curr_ratioconj in all_precisions:
            all_precisions[curr_ratioconj].append(curr_precisions)
        else:
            all_precisions[curr_ratioconj] = [curr_precisions]
    
    all_ratioconj = np.sort(np.unique(all_ratioconj))
    
    nb_experiments /= all_ratioconj.size

    nb_ratios = all_ratioconj.size

    # Now have to put everything in a nice 4D array...
    results_array = np.zeros((all_ratioconj.size, param1_space.size, param2_space.size, nb_repeats*nb_experiments))
    for ratioconj_i in np.arange(all_ratioconj.size):
        for par1 in np.arange(param1_space.size):
            for par2 in np.arange(param2_space.size):
                for exp in np.arange(nb_experiments):
                    try:
                        results_array[ratioconj_i, par1, par2, nb_repeats*(exp):nb_repeats*(exp+1)] = all_precisions[all_ratioconj[ratioconj_i]][exp][par1][par2]
                    except IndexError:
                        pass

    print "Size of results: %s" % results_array.shape.__str__()

    mean_invprecisions = np.mean(results_array, 3)
    mean_precisions = np.mean(1./results_array, 3)
    var_invprecisions = np.std(results_array, 3)
    var_precisions = np.std(1./results_array, 3)

    # Some quick/dirty plots
    if False:
        for ratioconj_i in np.arange(all_ratioconj.size):
            f = plt.figure()
            ax = f.add_subplot(111)
            im = ax.imshow(mean_precisions[ratioconj_i, :, 0:].T, origin='lower', aspect='auto')
            # im.set_extent((param1_space.min(), param1_space.max(), param2_space.min(), param2_space.max()))
            im.set_interpolation('nearest')
            # ax.xaxis.set_major_locator(plttic.NullLocator())
            # ax.yaxis.set_major_locator(plttic.NullLocator())
            plt.xticks(np.arange(param1_space.size), np.around(param1_space, 2), rotation=20)
            plt.yticks(np.arange((param2_space[0:]).size), np.around(param2_space[0:], 2))
            ax.set_xlabel('Scale of conjunctive cells')
            ax.set_ylabel('Scale of feature cells')
            f.colorbar(im)
            ax.set_title('Precision for Ratio conjunctive=%.3f' % all_ratioconj[ratioconj_i])
     
    if True:
           
        # Plots for abstract
        if False:
            optcurve = np.mean(np.min(mean_invprecisions,1),1)
            plot_mean_std_area(np.arange(nb_ratios), optcurve, np.std(np.std(mean_invprecisions-optcurve[:,np.newaxis, np.newaxis], 1),1))
            plt.title('Mean min invprecision, full')

            optcurve = np.min(np.min(mean_invprecisions,1),1)
            plot_mean_std_area(np.arange(nb_ratios), optcurve, np.std(np.std(mean_invprecisions - optcurve[:,np.newaxis, np.newaxis] ,1),1))
            plt.title('min min invprecision, full')

            optcurve = np.median(np.median(mean_invprecisions,1),1)
            plot_mean_std_area(np.arange(nb_ratios), optcurve, np.std(np.std(mean_invprecisions - optcurve[:,np.newaxis, np.newaxis] ,1),1))
            plt.title('median median invprecision, full')

            optcurve = np.mean(np.mean(mean_invprecisions,1),1)
            plot_mean_std_area(np.arange(nb_ratios), optcurve, np.std(np.std(mean_invprecisions - optcurve[:,np.newaxis, np.newaxis] ,1),1))
            plt.title('mean mean invprecision, full')

        if False:
            optcurve = np.mean(np.min(mean_invprecisions,1),1)
            plt.figure()
            plt.plot(np.arange(nb_ratios), optcurve)
            plt.title('Mean min invprecision, full')

            optcurve = np.min(np.min(mean_invprecisions,1),1)
            plt.figure()
            plt.plot(np.arange(nb_ratios), optcurve)
            plt.title('min min invprecision, full')

            optcurve = np.median(np.median(mean_invprecisions,1),1)
            plt.figure()
            plt.plot(np.arange(nb_ratios), optcurve)
            plt.title('median median invprecision, full')
            
            optcurve = np.median(np.min(mean_invprecisions,1),1)
            plt.figure()
            plt.plot(np.arange(nb_ratios), optcurve)
            plt.title('median min invprecision, full')

            optcurve = np.mean(np.mean(mean_invprecisions,1),1)
            plt.figure()
            plt.plot(np.arange(nb_ratios), optcurve)
            plt.title('mean mean invprecision, full')
        
        if False:
            # Those are wrong averages
            optcurve = np.mean(np.max(1./mean_invprecisions,1),1)
            plot_mean_std_area(np.arange(nb_ratios), optcurve, np.std(np.std(mean_invprecisions-optcurve[:,np.newaxis, np.newaxis], 1),1))
            plt.title('Mean max precision, full')

            optcurve = np.max(np.max(1./mean_invprecisions,1),1)
            plot_mean_std_area(np.arange(nb_ratios), optcurve, np.std(np.std(mean_invprecisions - optcurve[:,np.newaxis, np.newaxis] ,1),1))
            plt.title('max max precision, full')

            optcurve = np.median(np.median(1./mean_invprecisions,1),1)
            plot_mean_std_area(np.arange(nb_ratios), optcurve, np.std(np.std(mean_invprecisions - optcurve[:,np.newaxis, np.newaxis] ,1),1))
            plt.title('median median precision, full')
            
            optcurve = np.mean(np.mean(1./mean_invprecisions,1),1)
            plot_mean_std_area(np.arange(nb_ratios), optcurve, np.std(np.std(mean_invprecisions - optcurve[:,np.newaxis, np.newaxis] ,1),1))
            plt.title('mean mean precision, full')

        if False:
            optcurve = np.mean(np.max(mean_precisions,1),1)
            plot_mean_std_area(np.arange(nb_ratios), optcurve, np.std(np.std(mean_precisions-optcurve[:,np.newaxis, np.newaxis], 1),1))
            plt.title('Mean max precision, full')
            
            optcurve = np.median(np.median(mean_precisions,1),1)
            plot_mean_std_area(np.arange(nb_ratios), optcurve, np.std(np.std(mean_precisions - optcurve[:,np.newaxis, np.newaxis] ,1),1))
            plt.title('median median precision, full')
            
            optcurve = np.mean(np.mean(mean_precisions,1),1)
            plot_mean_std_area(np.arange(nb_ratios), optcurve, np.std(np.std(mean_precisions - optcurve[:,np.newaxis, np.newaxis] ,1),1))
            plt.title('mean mean precision, full')
            
            optcurve = np.max(np.max(mean_precisions,1),1)
            plot_mean_std_area(np.arange(nb_ratios), optcurve, np.std(np.std(mean_precisions - optcurve[:,np.newaxis, np.newaxis] ,1),1))
            plt.title('max max precision, full')


        if False:
            mean_precision_restr = mean_precisions[:,:,0:6]
            optcurve = np.mean(np.mean(mean_precision_restr,1),1)
            plot_mean_std_area(np.arange(nb_ratios), optcurve, np.std(np.std(mean_precision_restr - optcurve[:,np.newaxis, np.newaxis] ,1),1))
            plt.title('mean mean precision, restricted to lower half')
            optcurve = np.mean(np.max(mean_precision_restr,1),1)
            plot_mean_std_area(np.arange(nb_ratios), optcurve, np.std(np.std(mean_precision_restr-optcurve[:,np.newaxis, np.newaxis], 1),1))
            plt.title('mean max precision, restricted to lower half')
        
        stdcurve = np.zeros(all_ratioconj.size)

        # optcurve = np.mean(np.mean(mean_precisions,1),1)
        
        # if args.T ==1:
        #     optcurve = np.max(np.max(mean_precisions[:,:,4:-1],1),1) # works
        # elif args.T == 2:
        #     optcurve = np.max(np.max(mean_precisions[:,:,4:-1],1),1) # works
        # else:
        #     optcurve = np.max(np.max(mean_precisions[:,:,4:-1],1),1) # works
        # stdcurve = np.zeros(all_ratioconj.size)

        optcurve = np.max(np.max(1./mean_invprecisions[:, :, 0:], 1), 1)
        
        # # plot_mean_std_area(np.arange(nb_ratios), optcurve, np.std(np.std(mean_precisions - optcurve[:,np.newaxis, np.newaxis] ,1),1))
        # plt.title('mean mean precision, full')

        # optcurve = np.zeros(all_ratioconj.size)
        stdcurve = np.zeros(all_ratioconj.size)

        for r in np.arange(all_ratioconj.size):
        #     # indmax = argmax_indices(mean_precisions[r])
            indmax = argmax_indices(mean_precisions[r])
            # indmax = argmin_indices(mean_invprecisions[r])
            # optcurve[r] = mean_invprecisions[r, indmax[0], indmax[1]]
            # stdcurve[r] = var_invprecisions[r, indmax[0], indmax[1]]

            # optcurve[r] = mean_precisions[r, indmax[0], indmax[1]]
            # stdcurve[r] = var_precisions[r, indmax[0], indmax[1]]

        # optcurve = np.max(np.max(mean_precisions,1),1)
        # optcurve = np.max(np.max(1./mean_invprecisions,1),1)
        # stdcurve = np.std(np.std(mean_precisions - optcurve[:,np.newaxis, np.newaxis] ,1),1)

        # stdcurve /= np.max(optcurve)
        # optcurve /= np.max(optcurve)

        if should_plot:
            plot_mean_std_area(np.arange(nb_ratios), optcurve, stdcurve)
            plt.title('max max precision, full')
            plt.show()
        
    
    return locals()


def plot_effect_ratioconj():
    nb_ratios = 11
    T_max = 4
    all_optcurve = np.zeros((T_max, nb_ratios))
    all_optcurve_std = np.zeros((T_max, nb_ratios))

    for i in np.arange(T_max):
        all_vars = combine_mixed_two_scales(i, should_plot=False)
        all_optcurve[i] = all_vars['optcurve']
        all_optcurve_std[i] = all_vars['stdcurve']
        
    x = 0.1*np.arange(nb_ratios)
    f = plt.figure()
    ax = f.add_subplot(111)
    # ax = plot_mean_std_area(x, all_optcurve[0], all_optcurve_std[0])
    # for i in np.arange(1,3):
        # ax = plot_mean_std_area(x, all_optcurve[i], all_optcurve_std[i], ax_handle=ax)
    for i in np.arange(T_max):
        ax.plot(x, all_optcurve[i], linewidth=2)
    
    plt.rcParams['font.size'] = 17
    
    legends=['%d items' % (x+1) for x in np.arange(T_max)]
    legends[0] = '1 item'
    
    # plt.legend(legends, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=4, shadow=True)
    plt.legend(legends, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=4, fancybox=True, borderpad=0.3, columnspacing=0.5, borderaxespad=0.7, handletextpad=0, handlelength=1.5)
    # plt.xlabel('Ratio conjunctive/feature cells')
    # plt.ylim(0.0, 1.7)
    plt.ylim(0.0, 1.2)
    plt.yticks((0.25, 0.5, 0.75, 1.0))

    plt.show()


def combine_multiple_memory_curve():

    # all_output_files = glob.glob('Data/memory_curves/feat/feat_multiple_memories-*.npy*')
    # all_output_files = glob.glob('Data/memory_curves/conj/conj_multiple_memories-*.npy*')
    all_output_files = glob.glob('Data/memory_curves/cosyne_poster/conj_multiple_memories-*alpha0.89N*.npy*')
    # all_output_files = glob.glob('Data/memory_curves/cosyne_poster/sigmay0_1/conj_multiple_memories-*alpha0.92N*.npy*')
    
    assert len(all_output_files) > 0, "Wrong regular expression"

    # Iterate over them, load the corresponding signal and compute its metric
    all_precisions = []

    for curr_file in all_output_files:
        
        # Do a nice regular expression to catch the parameters and remove the useless random unique string
        # matched = re.search('^[a-zA-Z_\/0-9]*-T([0-9]*)alpha[0-9.]*M[0-9]*N[0-9.]*rcscale[0-9.]*sigmax[0-9.]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)multiple_memory_curve-([0-9a-z\-]*).npy', curr_file)
        matched = re.search('^[a-zA-Z_\/0-9]*-T([0-9]*)alpha[0-9.]*N[0-9.]*rcscale[0-9.]*sigmax[0-9.]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)multiple_memory_curve-([0-9a-z\-]*).npy', curr_file)
        # matched = re.search('^[a-zA-Z_\/0-9]*-T([0-9]*)alpha[0-9.]*N[0-9.]*rcscale[0-9.]*sigmax[0-9.]*sigmay[0-9.]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)multiple_memory_curve-([0-9a-z\-]*).npy', curr_file)

        # Get T
        T = int(matched.groups()[0])
                
        # Get nbrepeats and nbexperiments
        nb_repeats = int(matched.groups()[1])
        nb_experiments = int(len(all_output_files))

        print str(T)
        
        # Load the data
        curr_output = np.load(curr_file).item()
        
        # Reload some things
        args            = curr_output['args']
        curr_precisions = curr_output['all_precisions']

        # Store it
        all_precisions.append(curr_precisions)
        
    
    # nb_experiments /= all_ratioconj.size

    # Now have to put everything in a nice 4D array...
    results_array = np.zeros((T, T, nb_repeats*nb_experiments))
    for exp_i in np.arange(nb_experiments):
        try:
            results_array[:, :, nb_repeats*(exp_i):nb_repeats*(exp_i+1)] = all_precisions[exp_i]
        except IndexError:
            pass

    print "Size of results: %s" % results_array.shape.__str__()

    mean_precision = np.zeros((T,T))
    std_precision = np.zeros((T,T))
    for t1 in np.arange(T):
        for t2 in np.arange(T):
            precisions = 1./results_array[t1, t2]
            precisions[np.isinf(precisions)] = np.nan
            mean_precision[t1, t2]  = np.mean(precisions[~np.isnan(precisions)])
            std_precision[t1, t2]   = np.std(precisions[~np.isnan(precisions)])

    # Do a nanmean
    mean_results = 1./results_array
    mean_results[np.isinf(mean_results)] = np.nan

    tot_nonan= (mean_results.shape[2] - np.sum(np.isnan(mean_results), 2))
    mean_results = np.nansum(mean_results, axis=2)/tot_nonan


    plt.rcParams['font.size'] = 17
    
    f = plt.figure()
    ax = f.add_subplot(111)
    for t in np.arange(T):
        t_space_aligned_right = (T - np.arange(t+1))[::-1]
        # plot_mean_std_area(t_space_aligned_right, np.mean(1./all_precisions[t],1)[:t+1], np.std(1./all_precisions[t],1)[:t+1], ax_handle=ax)
        # semilogy_mean_std_area(t_space_aligned_right, np.mean(1./all_precisions[t],1)[:t+1], np.std(1./all_precisions[t],1)[:t+1], ax_handle=ax)
        # plt.semilogy(t_space_aligned_right, np.mean(1./results_array[t],1)[:t+1], 'o-')
        
        # plt.plot(t_space_aligned_right, np.mean(1./results_array[t],1)[:t+1], 'o-')
        plt.plot(t_space_aligned_right, mean_results[t, :t+1], 'o-', markersize=8, linewidth=2)
        # plot_mean_std_area(t_space_aligned_right, mean_precision[t, :t+1], std_precision[t, :t+1], ax_handle=ax)

    x_labels = ['-%d' % x for x in np.arange(T)[::-1]]
    x_labels[-1] = 'Last'

    ax.set_xticks(t_space_aligned_right)
    ax.set_xticklabels(x_labels)
    ax.set_xlim((0.8, T+0.2))
    ax.set_yticks((1, 2,3, 4,5))
    # ax.set_xlabel('Recall time')
    # ax.set_ylabel('Precision [rad]')
    plt.legend(['%d items' % (x+1) for x in np.arange(T)], loc='best', numpoints=1)

    return locals()




if __name__ == '__main__':
    # combine_plot_size_receptive_field_number_neurons()
    # all_vars = combine_mixed_two_scales(1)
    # all_vars = combine_mixed_two_scales(2)
    # all_vars = combine_mixed_two_scales(3)
    all_vars = combine_multiple_memory_curve()
    # plot_effect_ratioconj()
    

    plt.show()



