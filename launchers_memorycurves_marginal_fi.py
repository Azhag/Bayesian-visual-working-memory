#!/usr/bin/env python
# encoding: utf-8
"""
launchers_memorycurves_marginal_fi.py


Created by Loic Matthey on 2013-10-20
Copyright (c) 2013 . All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np

import utils
import dataio as DataIO
import progress

import load_experimental_data
import launchers


def load_prepare_datasets():
    data_gorgo11 = load_experimental_data.load_data_gorgo11(fit_mixture_model=True)
    gorgo11_T_space = data_gorgo11['data_to_fit']['n_items']
    gorgo11_emfits_meanstd = data_gorgo11['em_fits_nitems_arrays']
    gorgo11_emfits_meanstd['mean'][2, 0] = 0.0
    gorgo11_emfits_meanstd['std'][2, 0] = 0.0

    data_bays09 = load_experimental_data.load_data_bays09(fit_mixture_model=True)
    bays09_T_space = data_bays09['data_to_fit']['n_items']
    bays09_emfits_meanstd = data_bays09['em_fits_nitems_arrays']

    dict_experiments_to_plot = dict(bays09=dict(emfits=bays09_emfits_meanstd, T_space=bays09_T_space, axes=dict(ax_mem_plot_kappa=None, ax_em_plot_paper=None)), gorgo11=dict(emfits=gorgo11_emfits_meanstd, T_space=gorgo11_T_space, axes=dict(ax_mem_plot_kappa=None, ax_em_plot_paper=None)))
    plotting_parameters = dict(dict_experiments_to_plot=dict_experiments_to_plot)

    return plotting_parameters


def do_memory_plots(variables_launcher_running, plotting_parameters):
    dataio = variables_launcher_running.get('dataio', None)
    T_space = variables_launcher_running['T_space']

    dict_experiments_to_plot = plotting_parameters['dict_experiments_to_plot']
    suptitle_text = plotting_parameters.get('suptitle', '')
    reuse_axes = plotting_parameters.get('reuse_axes', True)

    result_em_fits_mean = utils.nanmean(variables_launcher_running['result_em_fits'], axis=-1)
    result_em_fits_std = utils.nanstd(variables_launcher_running['result_em_fits'], axis=-1)

    plt.ion()

    # Memory curve kappa
    def mem_plot_kappa(T_space_exp, exp_kappa_mean, exp_kappa_std=None, exp_name='', ax=None):

        if ax is not None:
            plt.figure(ax.get_figure().number)
            ax.hold(False)

        ax = utils.plot_mean_std_area(T_space_exp, exp_kappa_mean, exp_kappa_std, linewidth=3, fmt='o-', markersize=8, label='Experimental data', ax_handle=ax)

        ax.hold(True)

        ax = utils.plot_mean_std_area(T_space[:T_space_exp.max()], result_em_fits_mean[..., :T_space_exp.max(), 0], result_em_fits_std[..., :T_space_exp.max(), 0], xlabel='Number of items', ylabel="Memory error $[rad^{-2}]$", linewidth=3, fmt='o-', markersize=8, label='Fitted kappa', ax_handle=ax)

        if variables_launcher_running['all_parameters']['sigma_output'] > 0.0:
            ax.set_title("{{exp_name}} {M} {ratio_conj:.2f} {sigmax:.3f} {sigmay:.2f} {sigma_output:.2f}".format(**variables_launcher_running['all_parameters']).format(exp_name=exp_name))
        else:
            ax.set_title("{{exp_name}} {M} {ratio_conj:.2f} {sigmax:.3f} {sigmay:.3f}".format(**variables_launcher_running['all_parameters']).format(exp_name=exp_name))
        ax.legend()
        ax.set_xlim([0.9, T_space_exp.max()+0.1])
        ax.set_xticks(range(1, T_space_exp.max()+1))
        ax.set_xticklabels(range(1, T_space_exp.max()+1))

        if suptitle_text:
            ax.get_figure().suptitle(suptitle_text)

        ax.get_figure().canvas.draw()

        if dataio:
            if variables_launcher_running['all_parameters']['sigma_output'] > 0.0:
                dataio.save_current_figure('memorycurves%s_kappa_%s_M{M}_ratio{ratio_conj}_sigmax{sigmax}_sigmay{sigmay}_sigmaoutput{sigma_output}_{{label}}_{{unique_id}}.pdf'.format(**variables_launcher_running['all_parameters']) % ('_'.join(['', suptitle_text]*(suptitle_text != '')), exp_name))
            else:
                dataio.save_current_figure('memorycurves%s_kappa_%s_M{M}_ratio{ratio_conj}_sigmax{sigmax}_sigmay{sigmay}_{{label}}_{{unique_id}}.pdf'.format(**variables_launcher_running['all_parameters']) % ('_'.join(['', suptitle_text]*(suptitle_text != '')), exp_name))

        return ax

    # Plot EM Mixtures proportions
    def em_plot_paper(exp_name='', ax=None):

        if ax is None:
            _, ax = plt.subplots()

        if ax is not None:
            plt.figure(ax.get_figure().number)
            ax.hold(False)

        # mixture probabilities
        print result_em_fits_mean[..., 1]

        result_em_fits_mean[np.isnan(result_em_fits_mean)] = 0.0
        result_em_fits_std[np.isnan(result_em_fits_std)] = 0.0

        utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 1], result_em_fits_std[..., 1], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Target')
        ax.hold(True)
        utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 2], result_em_fits_std[..., 2], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Nontarget')
        utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 3], result_em_fits_std[..., 3], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Random')

        ax.legend(prop={'size':15})

        if variables_launcher_running['all_parameters']['sigma_output'] > 0.0:
            ax.set_title("{{exp_name}} {M} {ratio_conj:.2f} {sigmax:.3f} {sigmay:.2f} {sigma_output:.2f}".format(**variables_launcher_running['all_parameters']).format(exp_name=exp_name))
        else:
            ax.set_title("{{exp_name}} {M} {ratio_conj:.2f} {sigmax:.3f} {sigmay:.3f}".format(**variables_launcher_running['all_parameters']).format(exp_name=exp_name))

        ax.set_xlim([1.0, T_space.size])
        ax.set_ylim([0.0, 1.1])
        ax.set_xticks(range(1, T_space.size+1))
        ax.set_xticklabels(range(1, T_space.size+1))

        if suptitle_text:
            ax.get_figure().suptitle(suptitle_text)

        ax.get_figure().canvas.draw()

        if dataio:
            if variables_launcher_running['all_parameters']['sigma_output'] > 0.0:
                dataio.save_current_figure('memorycurves%s_emfits_%s_M{M}_ratio{ratio_conj}_sigmax{sigmax}_sigmay{sigmay}_sigmaoutput{sigma_output}_{{label}}_{{unique_id}}.pdf'.format(**variables_launcher_running['all_parameters']) % ('_'.join(['', suptitle_text]*(suptitle_text != '')), exp_name))
            else:
                dataio.save_current_figure('memorycurves%s_emfits_%s_M{M}_ratio{ratio_conj}_sigmax{sigmax}_sigmay{sigmay}_{{label}}_{{unique_id}}.pdf'.format(**variables_launcher_running['all_parameters']) % ('_'.join(['', suptitle_text]*(suptitle_text != '')), exp_name))

        return ax

    # Do all plots for all datasets
    for exper_name, exper_data in dict_experiments_to_plot.iteritems():
        ax1 = mem_plot_kappa(exper_data['T_space'], exper_data['emfits']['mean'][0], exper_data['emfits']['std'][0], exp_name=exper_name, ax=exper_data['axes']['ax_mem_plot_kappa'])
        ax2 = em_plot_paper(exp_name=exper_name, ax=exper_data['axes']['ax_em_plot_paper'])
        if reuse_axes:
            exper_data['axes']['ax_mem_plot_kappa'] = ax1
            exper_data['axes']['ax_em_plot_paper'] = ax2


def launcher_do_memory_curve_marginal_fi(args):
    '''
        Run the model for 1..T items, computing:
        - Precision of samples
        - EM mixture model fits
        - Marginal Inverse Fisher Information
    '''

    print "Doing a piece of work for launcher_do_memory_curve_marginal_fi"

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

    # Parameters to vary
    T_all = all_parameters['T']
    T_space = np.arange(1, T_all+1)

    # Result arrays
    result_all_precisions = np.nan*np.ones((T_space.size, all_parameters['num_repetitions']))
    result_marginal_inv_fi = np.nan*np.ones((T_space.size, 4, all_parameters['num_repetitions']))  # inv_FI, inv_FI_std, FI, FI_std
    result_em_fits = np.nan*np.ones((T_space.size, 5, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random, ll

    # If desired, will automatically save all Model responses.
    if all_parameters['collect_responses'] or all_parameters['subaction'] == 'collect_responses':
        result_responses = np.nan*np.ones((T_space.size, all_parameters['N'], all_parameters['num_repetitions']))
        result_target = np.nan*np.ones((T_space.size, all_parameters['N'], all_parameters['num_repetitions']))
        result_nontargets = np.nan*np.ones((T_space.size, all_parameters['N'], all_parameters['T']-1, all_parameters['num_repetitions']))

    search_progress = progress.Progress(T_space.size*all_parameters['num_repetitions'])

    for repet_i in xrange(all_parameters['num_repetitions']):
        for T_i, T in enumerate(T_space):
            print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

            print "Fit for T=%d, %d/%d" % (T, repet_i+1, all_parameters['num_repetitions'])

            # Update parameter
            all_parameters['T'] = T

            ### WORK WORK WORK work? ###

            # Fix some parameters
            # all_parameters['stimuli_generation'] = 'separated'
            # all_parameters['slice_width'] = np.pi/64.

            # Instantiate
            (_, _, _, sampler) = launchers.init_everything(all_parameters)

            # Sample
            sampler.run_inference(all_parameters)

            # Compute precision
            print "get precision..."
            result_all_precisions[T_i, repet_i] = sampler.get_precision()

            # Fit mixture model
            print "fit mixture model..."
            curr_params_fit = sampler.fit_mixture_model(use_all_targets=False)
            curr_params_fit['mixt_nontargets_sum'] = np.sum(curr_params_fit['mixt_nontargets'])
            result_em_fits[T_i, :, repet_i] = [curr_params_fit[key] for key in ('kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random', 'train_LL')]

            # Compute marginal inverse fisher info
            print "compute marginal inverse fisher info"
            marginal_fi_dict = sampler.estimate_marginal_inverse_fisher_info_montecarlo()
            result_marginal_inv_fi[T_i, :, repet_i] = [marginal_fi_dict[key] for key in ('inv_FI', 'inv_FI_std', 'FI', 'FI_std')]

            # If needed, store responses
            if all_parameters['subaction'] == 'collect_responses':
                (responses, target, nontarget) = sampler.collect_responses()
                result_responses[T_i, :, repet_i] = responses
                result_target[T_i, :, repet_i] = target
                result_nontargets[T_i, :, :T_i, repet_i] = nontarget

                print "collected responses"


            print result_all_precisions[T_i, repet_i], curr_params_fit, marginal_fi_dict

            ## Run callback function if exists
            if plots_during_simulation_callback:
                print "Doing plots..."
                try:
                    # Best super safe, if this fails then the simulation must continue!
                    plots_during_simulation_callback['function'](locals(), plots_during_simulation_callback['parameters'])
                    print "plots done."
                except:
                    print "error during plotting callback function", plots_during_simulation_callback['function'], plots_during_simulation_callback['parameters']

            ### /Work ###
            search_progress.increment()
            if run_counter % save_every == 0 or search_progress.done():
                dataio.save_variables_default(locals())
            run_counter += 1

    # Finished
    dataio.save_variables_default(locals())

    print "All finished"
    return locals()


def launcher_do_memory_curve_marginal_fi_withplots(args):
    '''
        Just like launcher_do_memory_curve_marginal_fi but do plots as well
    '''

    all_parameters = utils.argparse_2_dict(args)

    if all_parameters['code_type'] == 'hierarchical':
        # Use ratio_conj for plotting/titles
        if all_parameters['ratio_hierarchical'] is not None:
            all_parameters['ratio_conj'] = all_parameters['ratio_hierarchical']

    ### Load the experimental data for the plots
    plotting_parameters = load_prepare_datasets()

    ### Now do the plots
    if all_parameters.get('do_plots_during_simulation', False):
        # Define the callback function.
        all_parameters['plots_during_simulation_callback'] = dict(function=do_memory_plots, parameters=plotting_parameters)
        # Run the launcher_do_memory_curve_marginal_fi, plots are done during the runs automatically
        other_launcher_results = launcher_do_memory_curve_marginal_fi(all_parameters)
    else:
        # Run the launcher_do_memory_curve_marginal_fi, will do plots later
        other_launcher_results = launcher_do_memory_curve_marginal_fi(args)

        # Do the plots
        do_memory_plots(other_launcher_results, plotting_parameters)

    # Return the output of the other launcher.
    return other_launcher_results


def launcher_do_memory_curve_marginal_fi_withplots_live(args):
    '''
        Just like launcher_do_memory_curve_marginal_fi but do plots as well, while the simulation is going on
    '''

    all_parameters = utils.argparse_2_dict(args)

    all_parameters['do_plots_during_simulation'] = True

    return launcher_do_memory_curve_marginal_fi_withplots(all_parameters)

