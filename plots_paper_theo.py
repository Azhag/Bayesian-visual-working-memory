#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from datagenerator import *
from hierarchicalrandomnetwork import *
from randomfactorialnetwork import *
from statisticsmeasurer import *
from slicesampler import *
from utils import *
from dataio import *
from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *
from launchers import *

def do_plots_networks():
    if True:
        # Plot conj coverage for abstract
        R = 2
        M = 400
        MM = int(np.floor(M ** 0.5) ** 2.)
        plt.rcParams['font.size'] = 17
        # rn = RandomFactorialNetwork.create_full_conjunctive(M, R=R, scale_moments=(0.45, 0.001), ratio_moments=(1.0, 0.001))
        rn = RandomFactorialNetwork.create_full_conjunctive(M, R=R, scale_moments=(0.5, 0.2), ratio_moments=(1.0, 0.2))
        # ax = rn.plot_coverage_feature_space(nb_stddev=0.7)
        ax = rn.plot_coverage_feature_space(nb_stddev=0.7, alpha_ellipses=0.3, specific_neurons=np.arange(MM))
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
        ax.set_yticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))

    if True:
        # Plt feat coverage for abstract
        M = 50
        rn = RandomFactorialNetwork.create_full_features(M, R=R, scale=0.5, ratio=25, nb_feature_centers=1)
        # ax = rn.plot_coverage_feature_space(nb_stddev=0.7, alpha_ellipses=0.2)
        ax = rn.plot_coverage_feature_space(nb_stddev=0.7, alpha_ellipses=0.3, facecolor='r')
        boxsize = 4.4
        ax.set_xlim(-boxsize, boxsize)
        ax.set_ylim(-boxsize, boxsize)
        ax.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
        ax.set_yticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))

    if False:
        # Plot mixed coverage

        # %run experimentlauncher.py --code_type mixed --inference_method none --rc_scale 1.9 --rc_scale2 0.1 --feat_ratio -150
        ax = random_network.plot_coverage_feature_space(alpha_ellipses=0.2, specific_neurons=np.arange(100, 140, 2), facecolor='r', height_factor=2.5)
        ax = random_network.plot_coverage_feature_space(alpha_ellipses=0.2, specific_neurons=np.arange(220, 260, 2), facecolor='r', width_factor=2.5, ax=ax)
        ax = random_network.plot_coverage_feature_space(alpha_ellipses=0.2, specific_neurons=np.arange(60), facecolor='b', ax=ax)
        
        boxsize = np.pi*1.1
        ax.set_xlim(-boxsize, boxsize)
        ax.set_ylim(-boxsize, boxsize)
        

    plt.show()


def fisher_information_1obj_2d():
    # %run experimentlauncher.py --action_to_do launcher_do_fisher_information_estimation --subaction rcscale_dependence --M 100 --N 500 --sigmax 0.1 --sigmay 0.0001 --label fi_compare_paper --num_samples 100
    # Used the boxplot. And some
    
    plt.rcParams['font.size'] = 16

    plt.figure()
    b = plt.boxplot([FI_rc_curv_all[rc_scale_i], FI_rc_samples_all[rc_scale_i].flatten(), FI_rc_precision_all[rc_scale_i], FI_rc_theo_all[rc_scale_i, 0], FI_rc_theo_all[rc_scale_i, 1]])
    
    for key in ['medians', 'boxes', 'whiskers', 'caps']:
        for line in b[key]:
            line.set_linewidth(2)

    # plt.boxplot([FI_rc_curv_all[rc_scale_i], FI_rc_precision_all[rc_scale_i], FI_rc_theo_all[rc_scale_i, 0], FI_rc_theo_all[rc_scale_i, 1]])
    plt.title('Comparison Curvature vs samples estimate. Rscale: %d' % rc_scale)
    plt.xticks([1, 2, 3, 4, 5], ['Curvature', 'Samples', 'Precision', 'Theo', 'Theo large N'], rotation=45)
    # plt.xticks([1, 2, 3, 4], ['Curvature', 'Precision', 'Theo', 'Theo large N'], rotation=45)

    dataio.save_current_figure('FI_rc_comparison_curv_samples_%d-{label}_{unique_id}.pdf' % rc_scale)


def posterior_plots():
    '''
        Do the plots showing how the recall works.

        Put 3 objects, show the datapoint, the full posterior, the cued posterior and a sample from it
    '''

    # Conjunctive population
    all_parameters = dict(alpha=1.0, T=3, N=10, M=25**2, sigmay=0.001, sigmax=0.5, stimuli_generation='constant', R=2, rc_scale=5.0, rc_scale2=1, feat_ratio=20., autoset_parameters=True, code_type='conj')
    # all_parameters = dict(alpha=1.0, T=3, N=10, M=10**2, sigmay=0.001, sigmax=0.1, stimuli_generation='constant', R=2, rc_scale=5.0, feat_ratio=20., autoset_parameters=True, code_type='conj')

    (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)

    plt.rcParams['font.size'] = 18

    if False:
        data_gen.show_datapoint(n=1)
        sampler.plot_likelihood_variation_twoangles(n=1, interpolation='bilinear')
        sampler.plot_likelihood_correctlycuedtimes(n=1)
        sampler.plot_likelihood_correctlycuedtimes(n=1, should_exponentiate=True)

    # Feature population
    all_parameters['code_type'] = 'feat'
    all_parameters['M'] = 75*2
    all_parameters['sigmax'] = 0.1

    (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)

    # print random_network.neurons_sigma[0,0], random_network.neurons_sigma[0,1]

    if False:
        data_gen.show_datapoint(n=1)
        sampler.plot_likelihood_variation_twoangles(n=1, interpolation='bilinear')
        sampler.plot_likelihood_correctlycuedtimes(n=1)
        sampler.plot_likelihood_correctlycuedtimes(n=1, should_exponentiate=True)

    # Mixed population
    all_parameters['code_type'] = 'mixed'
    all_parameters['M'] = 200
    all_parameters['sigmax'] = 0.15
    all_parameters['rc_scale'] = 2.5
    all_parameters['rc_scale2'] = stddev_to_kappa(np.pi)
    all_parameters['ratio_conj'] = 0.5
    all_parameters['feat_ratio'] = stddev_to_kappa(2.*np.pi/int(all_parameters['M']*all_parameters['ratio_conj']/2.))/stddev_to_kappa(np.pi)
    
    (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)

    if True:
        # data_gen.show_datapoint(n=1)
        sampler.plot_likelihood_variation_twoangles(n=1, interpolation='bilinear')
        sampler.plot_likelihood_correctlycuedtimes(n=1)
        sampler.plot_likelihood_correctlycuedtimes(n=1, should_exponentiate=True)


    return locals()






if __name__ == '__main__':
     
    # do_plots_networks()

    all_vars = posterior_plots()

    variables_to_reinstantiate = ['data_gen', 'sampler', 'stat_meas', 'random_network', 'args', 'constrained_parameters', 'data_pbs', 'dataio']

    for var_reinst in variables_to_reinstantiate:
        if var_reinst in all_vars:
            vars()[var_reinst] = all_vars[var_reinst]

    plt.show()





