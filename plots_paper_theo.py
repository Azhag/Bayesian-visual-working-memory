#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from experimentlauncher import *

from datagenerator import *
from hierarchicalrandomnetwork import *
from randomfactorialnetwork import *
from statisticsmeasurer import *
from slicesampler import *
from utils import *
from dataio import *
from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *
from launchers import *

plt.rcParams['font.size'] = 17

def do_plots_population_codes():
    
    if False:
        # Plot conj coverage for abstract

        M = int(17**2.)
        rcscale = 8.
        # plt.rcParams['font.size'] = 17

        selected_neuron = M/2+3

        plt.ion()

        rn = RandomFactorialNetwork.create_full_conjunctive(M, rcscale=rcscale)
        ax = rn.plot_coverage_feature_space(alpha_ellipses=0.3, facecolor='b', lim_factor=1.1, nb_stddev=1.1)
        ax = rn.plot_coverage_feature_space(alpha_ellipses=0.3, facecolor='b', lim_factor=1.1, nb_stddev=1.1, specific_neurons=[selected_neuron], ax=ax)

        ax.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
        ax.set_yticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))

        ax.set_xlabel('')
        ax.set_ylabel('')

        ax.get_figure().canvas.draw()

        # To be run in ETS_TOOLKIT=qt4 mayavi2
        rn.plot_neuron_activity_3d(selected_neuron, precision=100, weight_deform=0.0, draw_colorbar=False)
        try:
            import mayavi.mlab as mplt

            mplt.view(0.0, 45.0, 45.0, [0., 0., 0.])
            mplt.draw()
        except:
            pass


    if True:
        # Plt feat coverage for abstract
        M = 50
        selected_neuron = M/4

        rn = RandomFactorialNetwork.create_full_features(M, scale=0.01, ratio=5000, nb_feature_centers=1)
        # rn = RandomFactorialNetwork.create_full_features(M, autoset_parameters=True, nb_feature_centers=1)
        # ax = rn.plot_coverage_feature_space(nb_stddev=0.7, alpha_ellipses=0.2)
        ax = rn.plot_coverage_feature_space(nb_stddev=2.0, alpha_ellipses=0.3, facecolor='r', lim_factor=1.1)
        ax = rn.plot_coverage_feature_space(nb_stddev=2.0, alpha_ellipses=0.4, facecolor='r', lim_factor=1.1, specific_neurons=[selected_neuron], ax=ax)
        
        ax.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
        ax.set_yticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))

        ax.set_xlabel('')
        ax.set_ylabel('')

        ax.get_figure().canvas.draw()

        if False:
            rn.plot_neuron_activity_3d(selected_neuron, precision=100, weight_deform=0.0, draw_colorbar=False)
            try:
                import mayavi.mlab as mplt

                mplt.view(0.0, 45.0, 45.0, [0., 0., 0.])
                mplt.draw()
            except:
                pass

    if False:
        # Plot mixed coverage

        autoset_parameters = False
        M = 300

        # %run experimentlauncher.py --code_type mixed --inference_method none --rc_scale 1.9 --rc_scale2 0.1 --feat_ratio -150
        conj_params = dict(scale_moments=[1.7, 0.001], ratio_moments=[1.0, 0.0001])
        feat_params = dict(scale=0.01, ratio=-8000, nb_feature_centers=1)

        rn = RandomFactorialNetwork.create_mixed(M, ratio_feature_conjunctive=0.2, conjunctive_parameters=conj_params, feature_parameters=feat_params, autoset_parameters=autoset_parameters)

        ax = rn.plot_coverage_feature_space(nb_stddev=2.0, alpha_ellipses=0.2, specific_neurons=np.arange(60, 180, 4), facecolor='r', lim_factor=1.1)
        ax = rn.plot_coverage_feature_space(nb_stddev=2.0, alpha_ellipses=0.2, specific_neurons=np.arange(180, 300, 4), facecolor='r', ax=ax, lim_factor=1.1)
        ax = rn.plot_coverage_feature_space(alpha_ellipses=0.2, specific_neurons=np.arange(60), facecolor='b', ax=ax, lim_factor=1.1)
        
        ax.set_xlabel('')
        ax.set_ylabel('')

        ax.get_figure().canvas.draw()

    if False:
        # Plot hierarchical coverage

        M = 100
        hrn_feat = HierarchialRandomNetwork(M, normalise_weights=1, type_layer_one='feature', optimal_coverage=True, M_layer_one=100, distribution_weights='exponential', threshold=1.0, output_both_layers=True)

        hrn_feat.plot_coverage_feature(nb_layer_two_neurons=3, facecolor_layerone='r', lim_factor=1.1)



    return locals()


def fisher_information_1obj_2d():
    # %run experimentlauncher.py --action_to_do launcher_do_fisher_information_estimation --subaction rcscale_dependence --M 100 --N 500 --sigmax 0.1 --sigmay 0.0001 --label fi_compare_paper --num_samples 100
    # Used the boxplot. And some
    
    # plt.rcParams['font.size'] = 16

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

    all_parameters['sigmax'] = 0.6

    (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)

    plt.rcParams['font.size'] = 18

    if True:
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

    if False:
        # data_gen.show_datapoint(n=1)
        sampler.plot_likelihood_variation_twoangles(n=1, interpolation='bilinear')
        sampler.plot_likelihood_correctlycuedtimes(n=1)
        sampler.plot_likelihood_correctlycuedtimes(n=1, should_exponentiate=True)


    return locals()


def compare_fishertheo_precision():
    '''
        Small try to compare the Fisher Info with the precision of samples,
        for different values of M/rc_scale for a conjunctive network

        (not sure if used in paper)
    '''

    arguments_dict = dict(action_to_do='launcher_do_compare_fisher_info_theo', N=500, sigmax=0.5, sigmay=0.0001, num_samples=100, label='sigmax{sigmax:.2f}', autoset_parameters=False)

    arguments_dict['do_precision'] = True

    # Run the Experiment
    experiment_launcher = ExperimentLauncher(run=True, arguments_dict=arguments_dict)

    return experiment_launcher.all_vars



if __name__ == '__main__':

    all_vars = {}

    all_vars = do_plots_population_codes()
    # all_vars = posterior_plots()
    # all_vars = compare_fishertheo_precision()

    variables_to_reinstantiate = ['data_gen', 'sampler', 'stat_meas', 'random_network', 'args', 'constrained_parameters', 'data_pbs', 'dataio']

    for var_reinst in variables_to_reinstantiate:
        if var_reinst in all_vars:
            vars()[var_reinst] = all_vars[var_reinst]

    plt.show()





