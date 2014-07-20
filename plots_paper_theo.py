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
from dataio import *
from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *
from launchers import *
import load_experimental_data

import cPickle as pickle

import utils

plt.rcParams['font.size'] = 17

set_colormap = plt.cm.cubehelix

def do_plots_population_codes():

    # plt.set_cmap('cubehelix')

    if True:
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

        set_colormap()

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

    if True:
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


def plot_distribution_errors():
    '''
        Plot for central + uniform bump
    '''

    dataio = DataIO(label='papertheo_histogram_nontargets')

    plt.rcParams['font.size'] = 18


    arguments_dict = dict(N=1000, sigmax=0.2, sigmay=0.0001, num_samples=500, burn_samples=500, autoset_parameters=True, M=100, code_type='conj', T=3, inference_method='sample', stimuli_generation='random', stimuli_generation_recall='random')
    # arguments_dict = dict(N=1000, sigmax=0.2, sigmay=0.0001, num_samples=500, burn_samples=500, autoset_parameters=True, M=100, code_type='conj', T=3, inference_method='sample', stimuli_generation='random', stimuli_generation_recall='random')

    # Run the Experiment
    experiment_launcher = ExperimentLauncher(run=True, arguments_dict=arguments_dict)

    # Plots
    experiment_launcher.all_vars['sampler'].plot_histogram_errors(bins=51)
    dataio.save_current_figure('papertheo_histogram_errorsM%dsigmax%.2fT%d.pdf' % tuple([arguments_dict[key] for key in ('M', 'sigmax', 'T')]))

    if arguments_dict['T'] > 1:
        experiment_launcher.all_vars['sampler'].plot_histogram_bias_nontarget(dataio=dataio)

    return locals()


def fisher_information_1obj_2d():
    # %run experimentlauncher.py --action_to_do launcher_do_fisher_information_estimation --subaction rcscale_dependence --M 100 --N 500 --sigmax 0.1 --sigmay 0.0001 --label fi_compare_paper --num_samples 100
    # fi_compare_paper-launcher_do_fisher_information_estimation-d563945d-4af3-4983-8250-09731352cbf9.npy
    # Used the boxplot. And some

    dataio = DataIO(label='papertheo_fisherinfo_1obj2d', calling_function='')

    plt.rcParams['font.size'] = 16

    # Do a boxplot
    # b = plt.boxplot([FI_rc_curv_all[0], FI_rc_samples_all[0].flatten(), FI_rc_precision_all[0], FI_rc_theo_all[0, 0], FI_rc_theo_all[0, 1]])
    # for key in ['medians', 'boxes', 'whiskers', 'caps']:
    #     for line in b[key]:
    #         line.set_linewidth(2)

    # Do a bar plot instead
    if True:
        FI_rc_curv_mean_rcscale = np.mean(FI_rc_curv_all, axis=-1)
        FI_rc_curv_std_rcscale = np.std(FI_rc_curv_all, axis=-1)
        FI_rc_samples_mean_rcscale = np.mean(FI_rc_samples_all.reshape((rcscale_space.size, FI_rc_samples_all.shape[1]*FI_rc_samples_all.shape[2])), axis=-1)
        FI_rc_samples_std_rcscale = np.std(FI_rc_samples_all.reshape((rcscale_space.size, FI_rc_samples_all.shape[1]*FI_rc_samples_all.shape[2])), axis=-1)

        rcscale_i = 4
        FI_rc_curv_mean = FI_rc_curv_mean_rcscale[rcscale_i]
        FI_rc_curv_std = FI_rc_curv_std_rcscale[rcscale_i]
        FI_rc_samples_mean = FI_rc_samples_mean_rcscale[rcscale_i]
        FI_rc_samples_std = FI_rc_samples_std_rcscale[rcscale_i]
        FI_rc_precision = FI_rc_precision_all[rcscale_i]
        FI_rc_theo = FI_rc_theo_all[rcscale_i, 0]
        FI_rc_theo_largen = FI_rc_theo_all[rcscale_i, 1]
    else:
        FI_rc_curv_mean = np.mean(FI_rc_curv_all)
        FI_rc_curv_std = np.std(FI_rc_curv_all)
        FI_rc_samples_mean = np.mean(FI_rc_samples_all)
        FI_rc_samples_std = np.std(FI_rc_samples_all)

    values_bars = np.array([FI_rc_precision, FI_rc_theo, FI_rc_theo_largen, FI_rc_samples_mean, FI_rc_curv_mean])
    values_bars_std = np.array([np.nan, np.nan, np.nan, FI_rc_samples_std, FI_rc_curv_std])

    # set_colormap = plt.cm.gnuplot
    color_gen = [set_colormap((i+0.1)/(float(len(values_bars))+0.1)) for i in xrange(len(values_bars))][::-1]

    bars_indices = np.arange(values_bars.size)
    width = 0.7

    ## Plot all as bars
    f, ax = plt.subplots(figsize=(10,6))

    for bar_i in xrange(values_bars.size):
        plt.bar(bars_indices[bar_i], values_bars[bar_i], width=width, color=color_gen[bar_i], zorder=2)
        plt.errorbar(bars_indices[bar_i] + width/2., values_bars[bar_i], yerr=values_bars_std[bar_i], ecolor='k', capsize=20, capthick=2, linewidth=2, zorder=3)

    # Add the precision bar times 2
    plt.bar(bars_indices[0], 2*values_bars[0], width=width, color=color_gen[0], alpha=0.5, hatch='/', linestyle='dashed', zorder=1)

    plt.xticks(bars_indices + width/2., ['Precision', 'Fisher Information', 'Fisher Information\n Large N', 'Samples', 'Curvature'], rotation=0)
    plt.xlim((-0.2, 5.))
    f.canvas.draw()
    plt.tight_layout()

    dataio.save_current_figure('FI_rc_comparison_curv_samples-papertheo-{label}_{unique_id}.pdf')


def posterior_plots():
    '''
        Do the plots showing how the recall works.

        Put 3 objects, show the datapoint, the full posterior, the cued posterior and a sample from it
    '''

    # Conjunctive population
    all_parameters = dict(alpha=1.0, T=3, N=10, M=25**2, sigmay=0.001, sigmax=0.5, stimuli_generation='constant', R=2, rc_scale=5.0, rc_scale2=1, feat_ratio=20., autoset_parameters=True, code_type='conj', enforce_first_stimulus=True, stimuli_generation_recall='random')
    # all_parameters = dict(alpha=1.0, T=3, N=10, M=10**2, sigmay=0.001, sigmax=0.1, stimuli_generation='constant', R=2, rc_scale=5.0, feat_ratio=20., autoset_parameters=True, code_type='conj')

    all_parameters['sigmax'] = 0.6


    plt.rcParams['font.size'] = 18

    if True:
        (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)

        data_gen.show_datapoint(n=1, colormap='gray')
        sampler.plot_likelihood_variation_twoangles(n=1, interpolation='bilinear', normalize=True, colormap='gray')
        # sampler.plot_likelihood_correctlycuedtimes(n=1)

        ax_handle = sampler.plot_likelihood_correctlycuedtimes(n=1, should_exponentiate=True, show_current_theta=False)

        ax_handle.set_yticks([])
        ax_handle.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax_handle.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=18)
        ax_handle.get_figure().canvas.draw()
        plt.tight_layout(pad=1.6)
        ax_handle.get_figure().canvas.draw()

    # Feature population
    all_parameters['code_type'] = 'feat'
    all_parameters['M'] = 75*2
    all_parameters['sigmax'] = 0.1


    # print random_network.neurons_sigma[0,0], random_network.neurons_sigma[0,1]

    if False:
        (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)

        data_gen.show_datapoint(n=1)
        sampler.plot_likelihood_variation_twoangles(n=1, interpolation='bilinear', normalize=True, colormap='gray')
        # sampler.plot_likelihood_correctlycuedtimes(n=1)
        ax_handle = sampler.plot_likelihood_correctlycuedtimes(n=1, should_exponentiate=True, show_current_theta=False)

        ax_handle.set_yticks([])
        ax_handle.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax_handle.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=18)
        ax_handle.get_figure().canvas.draw()
        plt.tight_layout(pad=1.6)
        ax_handle.get_figure().canvas.draw()

    # Mixed population
    all_parameters['code_type'] = 'mixed'
    all_parameters['M'] = 200
    all_parameters['autoset_parameters'] = True
    all_parameters['sigmax'] = 0.05
    all_parameters['rc_scale'] = 2.5
    all_parameters['rc_scale2'] = stddev_to_kappa(np.pi)
    all_parameters['ratio_conj'] = 0.5
    all_parameters['feat_ratio'] = stddev_to_kappa(2.*np.pi/int(all_parameters['M']*all_parameters['ratio_conj']/2.))/stddev_to_kappa(np.pi)


    if False:
        (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)

        data_gen.show_datapoint(n=1, colormap='gray')

        sampler.plot_likelihood_variation_twoangles(n=1, interpolation='bilinear', normalize=True, colormap='gray')

        # sampler.plot_likelihood_correctlycuedtimes(n=1)

        ax_handle = sampler.plot_likelihood_correctlycuedtimes(n=1, should_exponentiate=True, show_current_theta=False)

        ax_handle.set_yticks([])
        ax_handle.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax_handle.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=18)
        ax_handle.get_figure().canvas.draw()
        plt.tight_layout(pad=1.6)
        ax_handle.get_figure().canvas.draw()



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


def plot_experimental_mixture():
    '''
        Cheat and get data from Bays 2008 from figure...
    '''

    data_bays2009 = load_experimental_data.load_data_bays09(fit_mixture_model=True)
    experimental_mixtures_mean = data_bays2009['em_fits_nitems_arrays']['mean'][1:]
    experimental_mixtures_std = data_bays2009['em_fits_nitems_arrays']['std'][1:]
    experimental_mixtures_mean[np.isnan(experimental_mixtures_mean)] = 0.0
    experimental_mixtures_std[np.isnan(experimental_mixtures_std)] = 0.0

    experimental_mixtures_sem = experimental_mixtures_std/np.sqrt(np.unique(data_bays2009['subject']).size)
    items_space = np.unique(data_bays2009['n_items'])

    f, ax = plt.subplots()
    ax = plot_multiple_mean_std_area(items_space, experimental_mixtures_mean, experimental_mixtures_sem, ax_handle=ax, linewidth=2)

    ax.set_xlim((1.0, 5.0))
    ax.set_ylim((0.0, 1.1))
    # # ax.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
    ax.set_yticks((0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    ax.set_xticks((1, 2, 3, 4, 5))
    # plt.legend(['Target', 'Non-target', 'Random'], loc='upper right', fancybox=True, borderpad=0.3)

    return locals()


def plot_marginalfisherinfo_1d():
    N     = 50
    kappa = 6.0
    sigma = 0.5
    amplitude = 1.0
    min_distance = 0.0001

    dataio = DataIO(label='compute_fimarg', calling_function='')
    additional_comment = ''

    def population_code_response(theta, pref_angles=None, N=100, kappa=0.1, amplitude=1.0):
        if pref_angles is None:
            pref_angles = np.linspace(0., 2*np.pi, N, endpoint=False)

        return amplitude*np.exp(kappa*np.cos(theta - pref_angles))/(2.*np.pi*scsp.i0(kappa))

    pref_angles = np.linspace(-np.pi, np.pi, N, endpoint=False)

    ## Estimate likelihood
    num_points = 500
    # num_points_space = np.arange(50, 1000, 200)
    # effects_num_points = []

    # all_angles = np.linspace(0., 2.*np.pi, num_points, endpoint=False)
    all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)

    theta1_space = np.array([0.])
    theta2_space = all_angles

    def enforce_distance(theta1, theta2, min_distance=0.1):
        return np.abs(utils.wrap_angles(theta1 - theta2)) > min_distance


    min_distance_space = np.array([np.pi/30., np.pi/10., np.pi/4.])

    inv_FI_search = np.zeros((min_distance_space.size))
    FI_search = np.zeros((min_distance_space.size))
    FI_search_inv = np.zeros((min_distance_space.size))
    inv_FI_1_search = np.zeros((min_distance_space.size))
    inv_FI_search_full = np.zeros((min_distance_space.size, theta1_space.size, theta2_space.size))

    search_progress = progress.Progress(min_distance_space.size)

    for m, min_distance in enumerate(min_distance_space):

        if search_progress.percentage() % 5.0 < 0.0001:
            print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

        inv_FI_all = np.ones((theta1_space.size, theta2_space.size))*np.nan
        FI_all = np.ones((theta1_space.size, theta2_space.size, 2, 2))*np.nan
        inv_FI_1 = np.ones(theta1_space.size)*np.nan
        FI_all_inv = np.ones((theta1_space.size, theta2_space.size, 2, 2))*np.nan

        # Check inverse FI for given min_distance and kappa
        for i, theta1 in enumerate(theta1_space):
            der_1 = kappa*np.sin(pref_angles - theta1)*population_code_response(theta1, pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude)

            for j, theta2 in enumerate(theta2_space):

                if enforce_distance(theta1, theta2, min_distance=min_distance):
                    # Only compute if theta1 different enough of theta2

                    der_2 = kappa*np.sin(pref_angles - theta2)*population_code_response(theta2, pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude)

                    # FI for 2 objects
                    FI_all[i, j, 0, 0] = np.sum(der_1**2.)/(2.*sigma**2.)
                    FI_all[i, j, 0, 1] = np.sum(der_1*der_2)/(2.*sigma**2.)
                    FI_all[i, j, 1, 0] = np.sum(der_1*der_2)/(2.*sigma**2.)
                    FI_all[i, j, 1, 1] = np.sum(der_2**2.)/(2.*sigma**2.)
                    FI_all_inv[i, j] = np.linalg.inv(FI_all[i, j])

                    # Inv FI for 2 objects
                    inv_FI_all[i, j] = (2.*sigma**2.)/(np.sum(der_1**2.) - np.sum(der_1*der_2)**2./np.sum(der_2**2.))

            inv_FI_search_full[m, i] = inv_FI_all[i]

            # FI for 1 object
            inv_FI_1[i] = sigma**2./np.sum(der_1**2.)

        # inv_FI_search[m, k] = np.mean(inv_FI_all)
        inv_FI_search[m] = np.mean(np.ma.masked_invalid(inv_FI_all))
        FI_search[m] = np.mean(np.ma.masked_invalid(FI_all[..., 0, 0]))
        FI_search_inv[m] = np.mean(np.ma.masked_invalid(FI_all_inv[..., 0, 0]))

        inv_FI_1_search[m] = np.mean(inv_FI_1)

        search_progress.increment()

    print "FI_2obj_invtheo: ", inv_FI_search
    print "inv(FI_2obj_theo): ", FI_search_inv
    print "FI_2obj_theo[0,0]^-1 (wrong): ", 1./FI_search
    print "FI_1obj_theoinv: ", inv_FI_1_search
    print "2 obj effects: ", inv_FI_search/inv_FI_1_search

    plt.rcParams['font.size'] = 16

    left = 0.75
    f, axes = plt.subplots(ncols=min_distance_space.size)
    min_distance_labels = ['\\frac{\pi}{30}', '\\frac{\pi}{10}', '\\frac{\pi}{4}']
    titles_positions = [0.5, 0.4, 0.6]
    for m, min_distance in enumerate(min_distance_space):
        axes[m].bar(left/2., inv_FI_search[m]/inv_FI_1_search[0], width=left)
        axes[m].bar(left*2., inv_FI_1_search[m]/inv_FI_1_search[0], width=left, color='r')
        # axes[m].plot(left+np.arange(2), 0.5*inv_FI_search[m]*np.ones(2), 'r:')
        axes[m].set_xlim((0.0, left*3.5))
        # axes[m].set_ylim((0., 0.2))
        axes[m].set_xticks((left, left*5./2.))
        axes[m].set_xticklabels(("$\\tilde{I_F}^{-1}$", "${I_F^{(1)}}^{-1}$"))
        axes[m].set_yticks((0, 1, 2, 3))
        # axes[m].set_title('$min(\\theta_i - \\theta_j) = %s$' % min_distance_labels[m])
        axes[m].text(0.5, 1.05, '$min(\\theta_i - \\theta_j) = %s$' % min_distance_labels[m], transform=axes[m].transAxes, horizontalalignment='center', fontsize=18)

    # plt.figure()
    # plt.bar(xrange(3), np.array(zip(FI_search_inv, inv_FI_1_search)))

    # plt.figure()
    # plt.semilogy(min_distance_space, (inv_FI_search/inv_FI_1_search)[:, 1:], linewidth=2)
    # plt.plot(np.linspace(0.0, 1.6, 100), np.ones(100)*2.0, 'k:', linewidth=2)
    # plt.xlabel('Minimum distance')
    # plt.ylabel('$\hat{I_F}^{-1}/{I_F^{(1)}}^{-1}$')

    return locals()

def plot_marginal_fisher_info_2d():
    # RUN computations_marginalfisherinfo_marginalposterior_2d_nstim.
    # - First plots used for 2d plot. Min distance = 0.1
    #    inv_FI_2d_2obj_search_compute_fimarg_2dnstim_b6397e92-c114-4df3-b8c2-89ac97a7c88c.pdf
    # - Second plots used for bars:  min distance=  0.1, 5 objects.
    #    bars_IF_5obj_compute_fimarg_2dnstim_3133f906-3d89-4739-85ea-0311d7c1f951.pdf

    import computations_marginalfisherinfo_marginalposterior_2d_nstim

    computations_marginalfisherinfo_marginalposterior_2d_nstim.main(to_plot = [1])
    computations_marginalfisherinfo_marginalposterior_2d_nstim.main(to_plot = [2])

    return locals()


def plot_specific_stimuli():
    '''
        Plots on specific stimuli pattern

        Got Mixed and Hieararchical code.

        Done in reloader_specific_stimuli_mixed_sigmaxrangebis_191013.

        Ran this on top
        %run experimentlauncher.py --action_to_do launcher_do_mixed_special_stimuli_fixratio  --M 200 --sigmax 0.255 --sigmay 0.001 --autoset_parameters --T 3 --code_type mixed --ratio_conj 0.045 --num_repetitions 20 --N 200 --specific_stimuli_random_centers --enforce_min_distance 0.0809 --stimuli_generation specific_stimuli --label mixed_specific_stimuli_additional_run_paper
    '''

    # Additional plot
    data = np.load( '/Users/loicmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/Experiments/specific_stimuli/specific_stimuli_mixed_sigmaxmindistance_autoset_M200_repetitions10_sigmaxrangebis_191013_outputs/mixed_specific_stimuli_additional_run_paper-launcher_do_mixed_special_stimuli-080172fc-428e-4854-b3c2-d8292ecfac61.npy').item()

    ratio_space = data['ratio_space']
    result_all_precisions_mean = nanmean(data['result_all_precisions'], axis=-1)
    result_all_precisions_std = nanstd(data['result_all_precisions'], axis=-1)
    result_em_fits_mean = nanmean(data['result_em_fits'], axis=-1)
    result_em_fits_std = nanstd(data['result_em_fits'], axis=-1)
    result_em_kappastddev_mean = nanmean(kappa_to_stddev(data['result_em_fits'][:, 0]), axis=-1)
    result_em_kappastddev_std = nanstd(kappa_to_stddev(data['result_em_fits'][:, 0]), axis=-1)

    min_distance = 0.0809

    savefigs = True

    dataio = DataIO(output_folder='/Users/loicmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/Experiments/specific_stimuli/specific_stimuli_mixed_sigmaxmindistance_autoset_M200_repetitions10_sigmaxrangebis_191013_outputs/', label='global_plots_specificstimuli_mixed_repet20')


    # Plot precision
    plot_mean_std_area(ratio_space, result_all_precisions_mean, result_all_precisions_std) #, xlabel='Ratio conjunctivity', ylabel='Precision of recall')
    # plt.title('Min distance %.3f' % min_distance)
    plt.ylim([0, np.max(result_all_precisions_mean + result_all_precisions_std)])

    if savefigs:
        dataio.save_current_figure('mindist%.2f_precisionrecall_forpaper_{label}_{unique_id}.pdf' % min_distance)

    # Plot kappa fitted
    plot_mean_std_area(ratio_space, result_em_fits_mean[:, 0], result_em_fits_std[:, 0]) #, xlabel='Ratio conjunctivity', ylabel='Fitted kappa')
    # plt.title('Min distance %.3f' % min_distance)
    plt.ylim([-0.1, np.max(result_em_fits_mean[:, 0] + result_em_fits_std[:, 0])])
    if savefigs:
        dataio.save_current_figure('mindist%.2f_emkappa_forpaper_{label}_{unique_id}.pdf' % min_distance)

    # Plot kappa-stddev fitted. Easier to visualize
    plot_mean_std_area(ratio_space, result_em_kappastddev_mean, result_em_kappastddev_std) #, xlabel='Ratio conjunctivity', ylabel='Fitted kappa_stddev')
    # plt.title('Min distance %.3f' % min_distance)
    plt.ylim([0, 1.1*np.max(result_em_kappastddev_mean + result_em_kappastddev_std)])
    if savefigs:
        dataio.save_current_figure('mindist%.2f_emkappastddev_forpaper_{label}_{unique_id}.pdf' % min_distance)


    # Plot LLH
    plot_mean_std_area(ratio_space, result_em_fits_mean[:, -1], result_em_fits_std[:, -1]) #, xlabel='Ratio conjunctivity', ylabel='Loglikelihood of Mixture model fit')
    # plt.title('Min distance %.3f' % min_distance)
    if savefigs:
        dataio.save_current_figure('mindist%.2f_emllh_forpaper_{label}_{unique_id}.pdf' % min_distance)

    # Plot mixture parameters
    plot_multiple_mean_std_area(ratio_space, result_em_fits_mean[:, 1:4].T, result_em_fits_std[:, 1:4].T)
    # plt.legend("Target", "Non-target", "Random")
    plt.ylim([0.0, 1.1])
    if savefigs:
        dataio.save_current_figure('mindist%.2f_emprobs_forpaper_{label}_{unique_id}.pdf' % min_distance)

    return locals()


def plot_bootstrap_randomsamples():
    '''
        Do histograms with random samples from bootstrap nontarget estimates
    '''

    dataio = DataIO(label='plotpaper_bootstrap_randomized')

    nb_bootstrap_samples = 200
    use_precomputed = True

    angle_space = np.linspace(-np.pi, np.pi, 51)
    bins_center = angle_space[:-1] + np.diff(angle_space)[0]/2

    data_bays2009 = load_experimental_data.load_data_bays09(fit_mixture_model=True)

    ## Super long simulation, use precomputed data maybe?
    if use_precomputed:
        data = pickle.load(open('/Users/loicmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/Data/cache_randomized_bootstrap_samples_plots_paper_theo_plotbootstrapsamples/bootstrap_histo_katz.npy', 'r'))

        responses_resampled = data['responses_resampled']
        error_nontargets_resampled = data['error_nontargets_resampled']
        error_targets_resampled = data['error_targets_resampled']
        hist_cnts_nontarget_bootstraps_nitems = data['hist_cnts_nontarget_bootstraps_nitems']
        hist_cnts_target_bootstraps_nitems = data['hist_cnts_target_bootstraps_nitems']
    else:
        responses_resampled = np.empty((np.unique(data_bays2009['n_items']).size, nb_bootstrap_samples), dtype=np.object)
        error_nontargets_resampled = np.empty((np.unique(data_bays2009['n_items']).size, nb_bootstrap_samples), dtype=np.object)
        error_targets_resampled = np.empty((np.unique(data_bays2009['n_items']).size, nb_bootstrap_samples), dtype=np.object)
        hist_cnts_nontarget_bootstraps_nitems = np.empty((np.unique(data_bays2009['n_items']).size, nb_bootstrap_samples, angle_space.size - 1))*np.nan
        hist_cnts_target_bootstraps_nitems = np.empty((np.unique(data_bays2009['n_items']).size, nb_bootstrap_samples, angle_space.size - 1))*np.nan

        for n_items_i, n_items in enumerate(np.unique(data_bays2009['n_items'])):
            # Data collapsed accross subjects
            ids_filtered = (data_bays2009['n_items'] == n_items).flatten()

            if n_items > 1:

                # Get random bootstrap nontargets
                bootstrap_nontargets = utils.sample_angle(data_bays2009['item_angle'][ids_filtered, 1:n_items].shape + (nb_bootstrap_samples, ))

                # Compute associated EM fits
                bootstrap_results = []
                for bootstrap_i in progress.ProgressDisplay(np.arange(nb_bootstrap_samples), display=progress.SINGLE_LINE):

                    em_fit = em_circularmixture_allitems_uniquekappa.fit(data_bays2009['response'][ids_filtered, 0], data_bays2009['item_angle'][ids_filtered, 0], bootstrap_nontargets[..., bootstrap_i])

                    bootstrap_results.append(em_fit)

                    # Get EM samples
                    responses_resampled[n_items_i, bootstrap_i] = em_circularmixture_allitems_uniquekappa.sample_from_fit(em_fit, data_bays2009['item_angle'][ids_filtered, 0], bootstrap_nontargets[..., bootstrap_i])

                    # Compute the errors
                    error_nontargets_resampled[n_items_i, bootstrap_i] = utils.wrap_angles(responses_resampled[n_items_i, bootstrap_i][:, np.newaxis] - bootstrap_nontargets[..., bootstrap_i])
                    error_targets_resampled[n_items_i, bootstrap_i] = utils.wrap_angles(responses_resampled[n_items_i, bootstrap_i] - data_bays2009['item_angle'][ids_filtered, 0])

                    # Bin everything
                    hist_cnts_nontarget_bootstraps_nitems[n_items_i, bootstrap_i], x, bins = utils.histogram_binspace(utils.dropnan(error_nontargets_resampled[n_items_i, bootstrap_i]), bins=angle_space, norm='density')
                    hist_cnts_target_bootstraps_nitems[n_items_i, bootstrap_i], x, bins = utils.histogram_binspace(utils.dropnan(error_targets_resampled[n_items_i, bootstrap_i]), bins=angle_space, norm='density')

    # Now show average histogram
    hist_cnts_target_bootstraps_nitems_mean = np.mean(hist_cnts_target_bootstraps_nitems, axis=-2)
    hist_cnts_target_bootstraps_nitems_std = np.std(hist_cnts_target_bootstraps_nitems, axis=-2)
    hist_cnts_target_bootstraps_nitems_sem = hist_cnts_target_bootstraps_nitems_std/np.sqrt(hist_cnts_target_bootstraps_nitems.shape[1])

    hist_cnts_nontarget_bootstraps_nitems_mean = np.mean(hist_cnts_nontarget_bootstraps_nitems, axis=-2)
    hist_cnts_nontarget_bootstraps_nitems_std = np.std(hist_cnts_nontarget_bootstraps_nitems, axis=-2)
    hist_cnts_nontarget_bootstraps_nitems_sem = hist_cnts_nontarget_bootstraps_nitems_std/np.sqrt(hist_cnts_target_bootstraps_nitems.shape[1])

    f1, axes1 = plt.subplots(ncols=np.unique(data_bays2009['n_items']).size-1, figsize=((np.unique(data_bays2009['n_items']).size-1)*6, 6), sharey=True)
    for n_items_i, n_items in enumerate(np.unique(data_bays2009['n_items'])):
        if n_items>1:
            utils.plot_mean_std_area(bins_center, hist_cnts_nontarget_bootstraps_nitems_mean[n_items_i], hist_cnts_nontarget_bootstraps_nitems_sem[n_items_i], ax_handle=axes1[n_items_i-1], color='k')

            # Now add the Data histograms
            axes1[n_items_i-1].bar(bins_center, data_bays2009['hist_cnts_nontarget_nitems_stats']['mean'][n_items_i], width=2.*np.pi/(angle_space.size-1), align='center', yerr=data_bays2009['hist_cnts_nontarget_nitems_stats']['sem'][n_items_i])
            # axes4[n_items_i-1].set_title('N=%d' % n_items)
            axes1[n_items_i-1].set_xlim([bins_center[0]-np.pi/(angle_space.size-1), bins_center[-1]+np.pi/(angle_space.size-1)])

            # axes3[n_items_i-1].set_ylim([0., 2.0])
            axes1[n_items_i-1].set_xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
            axes1[n_items_i-1].set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=16)

            # axes1[n_items_i-1].bar(bins_center, hist_cnts_nontarget_bootstraps_nitems_mean[n_items_i], width=2.*np.pi/(angle_space.size-1), align='center', yerr=hist_cnts_nontarget_bootstraps_nitems_std[n_items_i])
            axes1[n_items_i-1].get_figure().canvas.draw()

    if dataio is not None:
        plt.tight_layout()
        dataio.save_current_figure("hist_error_nontarget_persubj_{label}_{unique_id}.pdf")


    if False:
        f2, axes2 = plt.subplots(ncols=np.unique(data_bays2009['n_items']).size-1, figsize=((np.unique(data_bays2009['n_items']).size-1)*6, 6), sharey=True)
        for n_items_i, n_items in enumerate(np.unique(data_bays2009['n_items'])):
            utils.plot_mean_std_area(bins_center, hist_cnts_target_bootstraps_nitems_mean[n_items_i], hist_cnts_target_bootstraps_nitems_std[n_items_i], ax_handle=axes2[n_items_i-1])
            # axes2[n_items_i-1].bar(bins_center, hist_cnts_target_bootstraps_nitems_mean[n_items_i], width=2.*np.pi/(angle_space.size-1), align='center', yerr=hist_cnts_target_bootstraps_nitems_std[n_items_i])

    return locals()



if __name__ == '__main__':

    all_vars = {}

    # all_vars = do_plots_population_codes()
    # all_vars = posterior_plots()
    # all_vars = fisher_information_1obj_2d()
    # all_vars = compare_fishertheo_precision()
    # all_vars = plot_experimental_mixture()
    # all_vars = plot_marginalfisherinfo_1d()
    # all_vars = plot_marginal_fisher_info_2d()
    # all_vars = plot_specific_stimuli()

    # all_vars = plot_distribution_errors()

    # all_vars = plot_bootstrap_randomsamples()

    if 'experiment_launcher' in all_vars:
        all_vars.update(all_vars['experiment_launcher'].all_vars)


    variables_to_reinstantiate = ['data_gen', 'sampler', 'stat_meas', 'random_network', 'args', 'constrained_parameters', 'data_pbs', 'dataio', 'experiment_launcher']

    for var_reinst in variables_to_reinstantiate:
        if var_reinst in all_vars:
            vars()[var_reinst] = all_vars[var_reinst]

    plt.show()





