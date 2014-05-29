#!/usr/bin/env python
# encoding: utf-8
"""
launchers_receptivesize.py

Created by Loic Matthey on 2014-05-20
Copyright (c) 2014 . All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np

import traceback

import utils
import dataio as DataIO
import progress

import launchers


def launcher_do_receptivesize_effect(args):
    '''
        Run the model for 1 item, varying the receptive size scale.
        Compute:
        - Precision of samples
        - EM mixture model fits
        - Marginal Inverse Fisher Information
    '''

    print "Doing a piece of work for launcher_do_receptivesize_effect"

    all_parameters = utils.argparse_2_dict(args)
    print all_parameters

    if all_parameters['burn_samples'] + all_parameters['num_samples'] < 200:
        print "WARNING> you do not have enough samples I think!", all_parameters['burn_samples'] + all_parameters['num_samples']

    if 'plots_during_simulation_callback' in all_parameters:
        plots_during_simulation_callback = all_parameters['plots_during_simulation_callback']
        del all_parameters['plots_during_simulation_callback']
    else:
        plots_during_simulation_callback = None

    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO.DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))
    save_every = 1
    run_counter = 0

    # Fix some parameters
    all_parameters['autoset_parameters'] = False
    all_parameters['feat_ratio'] = -1.  # hack to automatically set the ratio

    # Parameters to vary
    rcscale_space = np.linspace(0.0001, 40., 30)

    # Result arrays
    result_all_precisions = np.nan*np.ones((rcscale_space.size, all_parameters['num_repetitions']))
    result_marginal_inv_fi = np.nan*np.ones((rcscale_space.size, 4, all_parameters['num_repetitions']))  # inv_FI, inv_FI_std, FI, FI_std
    result_em_fits = np.nan*np.ones((rcscale_space.size, 5, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random, ll

    search_progress = progress.Progress(rcscale_space.size*all_parameters['num_repetitions'])

    for repet_i in xrange(all_parameters['num_repetitions']):
        for rc_scale_i, rc_scale in enumerate(rcscale_space):
            print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

            print "Fit for rc_scale=%.2f, %d/%d" % (rc_scale, repet_i+1, all_parameters['num_repetitions'])

            # Update parameter
            all_parameters['rc_scale'] = rc_scale

            ### WORK WORK WORK work? ###

            # Instantiate
            (_, _, _, sampler) = launchers.init_everything(all_parameters)

            # Sample
            sampler.run_inference(all_parameters)

            # Compute precision
            print "get precision..."
            result_all_precisions[rc_scale_i, repet_i] = sampler.get_precision()

            # Fit mixture model
            print "fit mixture model..."
            curr_params_fit = sampler.fit_mixture_model(use_all_targets=False)
            result_em_fits[rc_scale_i, :, repet_i] = [curr_params_fit[key] for key in ('kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random', 'train_LL')]

            # Compute marginal inverse fisher info
            print "compute marginal inverse fisher info"
            marginal_fi_dict = sampler.estimate_marginal_inverse_fisher_info_montecarlo()
            result_marginal_inv_fi[rc_scale_i, :, repet_i] = [marginal_fi_dict[key] for key in ('inv_FI', 'inv_FI_std', 'FI', 'FI_std')]


            print result_all_precisions[rc_scale_i, repet_i], curr_params_fit, marginal_fi_dict

            ## Run callback function if exists
            if plots_during_simulation_callback:
                print "Doing plots..."
                try:
                    # Best super safe, if this fails then the simulation must continue!
                    plots_during_simulation_callback['function'](locals(), plots_during_simulation_callback['parameters'])
                    print "plots done."
                except Exception as e:
                    print "error during plotting callback function", plots_during_simulation_callback['function'], plots_during_simulation_callback['parameters']
                    print e
                    traceback.print_exc()

            ### /Work ###
            search_progress.increment()
            if run_counter % save_every == 0 or search_progress.done():
                dataio.save_variables_default(locals())
            run_counter += 1

    # Finished
    dataio.save_variables_default(locals())

    print "All finished"
    return locals()



def receptivesize_effect_plots(variables_launcher_running, plotting_parameters):
    '''
        Do some plots (possibly live) with outputs from launcher_do_receptivesize_effect

    '''

    ### Load the experimental data for the plots
    # data_gorgo11 = load_experimental_data.load_data_gorgo11(fit_mixture_model=True)
    # gorgo11_T_space = data_gorgo11['data_to_fit']['n_items']
    # gorgo11_emfits_meanstd = data_gorgo11['em_fits_nitems_arrays']
    # gorgo11_emfits_meanstd['mean'][2, 0] = 0.0
    # gorgo11_emfits_meanstd['std'][2, 0] = 0.0

    # data_bays09 = load_experimental_data.load_data_bays09(fit_mixture_model=True)
    # bays09_T_space = data_bays09['data_to_fit']['n_items']
    # bays09_emfits_meanstd = data_bays09['em_fits_nitems_arrays']

    # Compute the "optimal" rcscale
    if variables_launcher_running['all_parameters']['code_type'] == 'conj':
        optimal_scale = utils.stddev_to_kappa(2.*np.pi/int(variables_launcher_running['all_parameters']['M']**0.5))
        optimal_scale_corrected = utils.stddev_to_kappa(np.pi/(int(variables_launcher_running['all_parameters']['M']**0.5)))
    elif variables_launcher_running['all_parameters']['code_type'] == 'feat':
        optimal_scale = utils.stddev_to_kappa(2.*np.pi/int(variables_launcher_running['all_parameters']['M']/2.))
        optimal_scale_corrected = utils.stddev_to_kappa(np.pi/int(variables_launcher_running['all_parameters']['M']/2.))
    else:
        optimal_scale = 0.0

    ### Now do the plots
    current_axes = plotting_parameters['axes']
    dataio = variables_launcher_running['dataio']
    rcscale_space = variables_launcher_running['rcscale_space']

    result_precision_stats = utils.nanstats(variables_launcher_running['result_all_precisions'], axis=-1)
    result_em_fits_stats = utils.nanstats(variables_launcher_running['result_em_fits'], axis=-1)
    result_marginal_fi_stats = utils.nanstats(variables_launcher_running['result_marginal_inv_fi'][:, 2], axis=-1)

    plt.ion()

    # Precision wrt rcscale_space
    def plot_precision_rcscale(ax=None):
        if ax is not None:
            plt.figure(ax.get_figure().number)
            ax.hold(False)

        # Curve of precision evolution.
        ax = utils.plot_mean_std_area(rcscale_space, result_precision_stats['mean'], result_precision_stats['std'], linewidth=3, fmt='o-', markersize=8, label='Precision', ax_handle=ax)

        ax.hold(True)

        ax.axvline(x=optimal_scale, color='r', linewidth=3)
        ax.axvline(x=optimal_scale_corrected, color='k', linewidth=3)

        ax.legend()
        ax.set_title("Precision {code_type} {M} {sigmax:.3f} {sigmay:.2f}".format(**variables_launcher_running['all_parameters']))


        ax.set_xlim(rcscale_space.min(), rcscale_space.max())
        ax.set_ylim(bottom=0.0)

        ax.get_figure().canvas.draw()

        dataio.save_current_figure('precision_rcscale_{code_type}_M{M}_sigmax{sigmax}_sigmay{sigmay}_{{label}}_{{unique_id}}.pdf'.format(**variables_launcher_running['all_parameters']))

        return ax

    # Precision wrt rcscale_space
    def plot_fisherinfo_rcscale(ax=None):
        if ax is not None:
            plt.figure(ax.get_figure().number)
            ax.hold(False)

        # Curve of precision evolution.
        ax = utils.plot_mean_std_area(rcscale_space, result_marginal_fi_stats['mean'], result_marginal_fi_stats['std'], linewidth=3, fmt='o-', markersize=8, label='Fisher info', ax_handle=ax)

        ax.hold(True)

        ax.axvline(x=optimal_scale, color='r', linewidth=3)
        ax.axvline(x=optimal_scale_corrected, color='k', linewidth=3)

        ax.legend()
        ax.set_title("FI {code_type} {M} {sigmax:.3f} {sigmay:.2f}".format(**variables_launcher_running['all_parameters']))

        ax.set_xlim(rcscale_space.min(), rcscale_space.max())
        ax.set_ylim(bottom=0.0)
        ax.get_figure().canvas.draw()

        dataio.save_current_figure('fi_rcscale_{code_type}_M{M}_sigmax{sigmax}_sigmay{sigmay}_{{label}}_{{unique_id}}.pdf'.format(**variables_launcher_running['all_parameters']))

        return ax

    # Memory curve kappa
    def plot_kappa_rcscale(ax=None):

        if ax is not None:
            plt.figure(ax.get_figure().number)
            ax.hold(False)

        ax = utils.plot_mean_std_area(rcscale_space, result_em_fits_stats['mean'][..., 0], result_em_fits_stats['std'][..., 0], linewidth=3, fmt='o-', markersize=8, label='Memory error $[rad^{-2}]$', ax_handle=ax)

        ax.hold(True)

        ax.axvline(x=optimal_scale, color='r', linewidth=3)
        ax.axvline(x=optimal_scale_corrected, color='k', linewidth=3)

        ax.legend()
        ax.set_xlim(rcscale_space.min(), rcscale_space.max())
        ax.set_ylim(bottom=0.0)

        ax.get_figure().canvas.draw()

        ax.set_title("kappa {code_type} {M} {sigmax:.3f} {sigmay:.2f}".format(**variables_launcher_running['all_parameters']))

        dataio.save_current_figure('kappa_rcscale_{code_type}_M{M}_sigmax{sigmax}_sigmay{sigmay}_{{label}}_{{unique_id}}.pdf'.format(**variables_launcher_running['all_parameters']))

        return ax

    # Plot EM Mixtures proportions
    def plot_mixtures_rcscale(ax=None):

        if ax is None:
            _, ax = plt.subplots()

        if ax is not None:
            plt.figure(ax.get_figure().number)
            ax.hold(False)

        result_em_fits_stats['mean'][np.isnan(result_em_fits_stats['mean'])] = 0.0
        result_em_fits_stats['std'][np.isnan(result_em_fits_stats['std'])] = 0.0

        utils.plot_mean_std_area(rcscale_space, result_em_fits_stats['mean'][..., 1], result_em_fits_stats['std'][..., 1], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Target')
        ax.hold(True)
        utils.plot_mean_std_area(rcscale_space, result_em_fits_stats['mean'][..., 2], result_em_fits_stats['std'][..., 2], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Nontarget')
        utils.plot_mean_std_area(rcscale_space, result_em_fits_stats['mean'][..., 3], result_em_fits_stats['std'][..., 3], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Random')

        ax.axvline(x=optimal_scale, color='r', linewidth=3)
        ax.axvline(x=optimal_scale_corrected, color='k', linewidth=3)

        ax.set_xlim(rcscale_space.min(), rcscale_space.max())
        ax.set_ylim(bottom=0.0, top=1.0)

        ax.legend(prop={'size':15})
        ax.set_title("mixts {code_type} {M} {sigmax:.3f} {sigmay:.2f}".format(**variables_launcher_running['all_parameters']))
        ax.get_figure().canvas.draw()

        dataio.save_current_figure('em_mixts_rcscale_{code_type}_M{M}_sigmax{sigmax}_sigmay{sigmay}_{{label}}_{{unique_id}}.pdf'.format(**variables_launcher_running['all_parameters']))

        return ax

    # Do all plots for all datasets
    current_axes['precision_rcscale'] = plot_precision_rcscale(ax=current_axes.get('precision_rcscale', None))
    current_axes['fisherinfo_rcscale'] = plot_fisherinfo_rcscale(ax=current_axes.get('fisherinfo_rcscale', None))
    current_axes['kappa_rcscale'] = plot_kappa_rcscale(ax=current_axes.get('kappa_rcscale', None))
    current_axes['mixtures_rcscale'] = plot_mixtures_rcscale(ax=current_axes.get('mixtures_rcscale', None))


def launcher_do_receptivesize_effect_withplots_live(args):
    '''
        Just like launcher_do_receptivesize_effect but do plots as well, while the simulation is going on
    '''

    all_parameters = utils.argparse_2_dict(args)

    plotting_parameters = dict(axes={})

    if all_parameters['plot_while_running']:
        # Define the callback function.
        all_parameters['plots_during_simulation_callback'] = dict(function=receptivesize_effect_plots, parameters=plotting_parameters)

        other_launcher_results = launcher_do_receptivesize_effect(all_parameters)
    else:
        # Run the launcher_do_memory_curve_marginal_fi, will do plots later
        other_launcher_results = launcher_do_receptivesize_effect(args)

        # Do the plots
        receptivesize_effect_plots(other_launcher_results, plotting_parameters)


    return other_launcher_results



