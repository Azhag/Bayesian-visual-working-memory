#!/usr/bin/env python
# encoding: utf-8

from datagenerator import *
from randomnetwork import *
from randomfactorialnetwork import *
from statisticsmeasurer import *
from slicesampler import *
from utils import *

import numpy as np
import pylab as plt

import scipy.special as scsp
import scipy.optimize as spopt
import scipy.interpolate as spint
import os.path
from matplotlib.patches import Ellipse
import glob
import re


def do_plots_networks():
    if True:
        # Plot conj coverage for abstract
        R = 2
        M = 400
        MM = int(np.floor(M**0.5)**2.)
        plt.rcParams['font.size'] = 17
        # rn = RandomFactorialNetwork.create_full_conjunctive(M, R=R, scale_moments=(0.45, 0.001), ratio_moments=(1.0, 0.001))
        rn = RandomFactorialNetwork.create_full_conjunctive(M, R=R, scale_moments=(0.5, 0.2), ratio_moments=(1.0, 0.2))
        # ax = rn.plot_coverage_feature_space(nb_stddev=0.7)
        ax = rn.plot_coverage_feature_space(nb_stddev=0.7, alpha_ellipses=0.3, specified_neurons=np.arange(MM))
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
        ax.set_yticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
        ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))

    if True:
        # Plt feat coverage for abstract
        M = 50
        rn = RandomFactorialNetwork.create_full_features(M, R=R,scale=0.5, ratio=25)
        # ax = rn.plot_coverage_feature_space(nb_stddev=0.7, alpha_ellipses=0.2)
        ax = rn.plot_coverage_feature_space(nb_stddev=0.7, alpha_ellipses=0.3, facecolor='r')
        boxsize=4.4
        ax.set_xlim(-boxsize, boxsize)
        ax.set_ylim(-boxsize, boxsize)
        ax.set_xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
        ax.set_yticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
        ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
    
    if True:
        # Plot mixed coverage for abstract
        M = 100
        conj_params = dict(scale_moments=(15., 0.001), ratio_moments=(1.0, 0.2))
        feat_params = dict(scale=0.3, ratio=40.)
        ratio_conj  = 0.1
        MM = int(np.floor((M*ratio_conj)**0.5)**2.)

        rn = RandomFactorialNetwork.create_mixed(M, R=R, ratio_feature_conjunctive=ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
        ##
        # ax = rn.plot_coverage_feature_space(nb_stddev=0.7, alpha_ellipses=0.25)
        ##
        ax = rn.plot_coverage_feature_space(nb_stddev=0.7, alpha_ellipses=0.2, specified_neurons=np.arange(MM), facecolor='b')
        ax = rn.plot_coverage_feature_space(nb_stddev=0.7, alpha_ellipses=0.35, specified_neurons=np.arange(M*ratio_conj,M, dtype='int'), facecolor='r', ax=ax)
        ##
        # ax = rn.plot_coverage_feature_space(nb_stddev=0.7, alpha_ellipses=0.2, specified_neurons=np.arange(MM))
        # ax = rn.plot_coverage_feature_space(nb_stddev=0.7, alpha_ellipses=0.3, specified_neurons=np.arange(M*ratio_conj,M, dtype='int'), ax=ax)

    boxsize=4.4
    ax.set_xlim(-boxsize, boxsize)
    ax.set_ylim(-boxsize, boxsize)
    ax.set_xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
    ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
    ax.set_yticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
    ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.show()


def do_network_output():
    N = 100
    T = 3
    K = 30
    M = 300
    R = 2
    

    random_network = RandomFactorialNetwork.create_full_conjunctive(M, R=R, scale_moments=(0.5, 0.01), ratio_moments=(1.0, 0.05))
    
    data_gen = DataGeneratorRFN(N, T, random_network, sigma_y = 0.02, sigma_x = 0.1, time_weights_parameters = dict(weighting_alpha=0.9, weighting_beta = 1.0, specific_weighting = 0.2, weight_prior='uniform'))

    # Find a good datapoint.
    data_gen.show_datapoint(n=2)


def do_plot_effect_conj():
    def combine_mixed_two_scales(data_to_use = 0, should_plot=True):

        # T1
        if data_to_use == 0:
            params = dict(label='T1_inabstract', files='Data/mixed_ratio_2scales_search/T1_N300_K100/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*K[0-9]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')
        # T2
        if data_to_use == 1:
            params = dict(label='T2_inabstract', files='Data/mixed_ratio_2scales_search/T2_N300_K100/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*K[0-9]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')
        
        # T3
        if data_to_use == 2:
            params = dict(label='T3_inabstract', files='Data/mixed_ratio_2scales_search/T3/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*K[0-9]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')
        
        # T4
        if data_to_use == 3:
            params = dict(label='T4_inabstract', files='Data/mixed_ratio_2scales_search/T4/ratioconj_2scales*.npy*', regexp='^[a-zA-Z_\/0-9]*-ratioconj([0-9.]*)T[0-9]*alpha[0-9.]*N[0-9.]*K[0-9]*numsamples[0-9]*nbrepeats([0-9.]*)nbexperiments([0-9.]*)mixed_two_scales-([0-9a-z\-]*).npy')
        

        # Get all output files
        all_output_files = glob.glob(params['files'])
        
        assert len(all_output_files) > 0, "Wrong regular expression"

        # Iterate over them, load the corresponding signal and compute its metric
        all_ratioconj = []
        all_precisions = {}

        for curr_file in all_output_files:
            
            # Do a nice regular expression to catch the parameters and remove the useless random unique string
            matched = re.search(params['regexp'], curr_file)
                
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
        # var_precisions = np.std(1./results_array, 3)

        stdcurve = np.zeros(all_ratioconj.size)
        # optcurve = np.max(np.max(1./mean_invprecisions[:,:,4:],1),1)
        optcurve = np.max(np.max(1./mean_invprecisions[:,:,0:],1),1)
    
        stdcurve /= np.max(optcurve)
        optcurve /= np.max(optcurve)
    
        return (optcurve, stdcurve)


    nb_ratios = 11
    T_max = 4
    all_optcurve = np.zeros((T_max, nb_ratios))
    all_optcurve_std = np.zeros((T_max, nb_ratios))

    for i in np.arange(T_max):
        (all_optcurve[i], all_optcurve_std[i]) = combine_mixed_two_scales(i, should_plot=False)
        
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

    

def plot_all_memory_curves():
     #%run gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork.py --action_to_do plot_multiple_memory_curve --input_filename /Users/loicmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/Data/all_memories_conj_T5_alpha09_sigmax01_r30_memory_curve-dea28e17-7330-43a2-bda1-f54aa2f100c4.npy
     pass

