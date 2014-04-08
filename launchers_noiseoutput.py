#!/usr/bin/env python
# encoding: utf-8
"""
launchers_noiseoutput.py


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


def launcher_do_noise_output_effect(args):
    '''
        Run the model for T items, with varying amount of sigma_output

        Check how the Mixture model parameters vary
    '''

    print "Doing a piece of work for launcher_do_noise_output_effect"

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
    precision_sigmaoutput = 5
    sigmaoutput_space = np.linspace(0.0, 2.0, precision_sigmaoutput)

    # Result arrays
    result_all_precisions = np.nan*np.ones((sigmaoutput_space.size, all_parameters['num_repetitions']))
    result_em_fits = np.nan*np.ones((sigmaoutput_space.size, 6, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random, ll, bic

    # If desired, will automatically save all Model responses.
    if all_parameters['collect_responses']:
        result_responses = np.nan*np.ones((sigmaoutput_space.size, all_parameters['N'], all_parameters['num_repetitions']))
        result_target = np.nan*np.ones((sigmaoutput_space.size, all_parameters['N'], all_parameters['num_repetitions']))
        result_nontargets = np.nan*np.ones((sigmaoutput_space.size, all_parameters['N'], all_parameters['T']-1, all_parameters['num_repetitions']))

    search_progress = progress.Progress(sigmaoutput_space.size*all_parameters['num_repetitions'])

    for repet_i in xrange(all_parameters['num_repetitions']):
        for sigmaoutput_i, sigma_output in enumerate(sigmaoutput_space):
            print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

            print "Fit for sigma_output=%.3f, %d/%d" % (sigma_output, repet_i+1, all_parameters['num_repetitions'])

            # Update parameter
            all_parameters['sigma_output'] = sigma_output

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
            result_all_precisions[sigmaoutput_i, repet_i] = sampler.get_precision()

            # Fit mixture model
            print "fit mixture model..."
            curr_params_fit = sampler.fit_mixture_model(use_all_targets=False)
            result_em_fits[sigmaoutput_i, :, repet_i] = [curr_params_fit[key] for key in ('kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random', 'train_LL', 'bic')]

            # If needed, store responses
            if all_parameters['collect_responses']:
                (responses, target, nontarget) = sampler.collect_responses()
                result_responses[sigmaoutput_i, :, repet_i] = responses
                result_target[sigmaoutput_i, :, repet_i] = target
                result_nontargets[sigmaoutput_i, :, :(all_parameters['T']-1), repet_i] = nontarget

                print "collected responses"


            print result_all_precisions[sigmaoutput_i, repet_i], curr_params_fit

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


def launcher_do_noise_output_effect_withplots(args):
    '''
        Just like launcher_do_noise_output_effect but do plots as well
    '''

    all_parameters = utils.argparse_2_dict(args)

    if all_parameters['code_type'] == 'hierarchical':
        # Use ratio_conj for plotting/titles
        if all_parameters['ratio_hierarchical'] is not None:
            all_parameters['ratio_conj'] = all_parameters['ratio_hierarchical']

    plotting_parameters = dict(axes=dict(ax_sigmaoutput_kappa=None, ax_sigmaoutput_mixtures=None))

    ### Now do the plots
    def do_sigma_output_plot(variables_launcher_running, plotting_parameters):
        dataio = variables_launcher_running['dataio']
        sigmaoutput_space = variables_launcher_running['sigmaoutput_space']

        result_em_fits_mean = utils.nanmean(variables_launcher_running['result_em_fits'], axis=-1)
        result_em_fits_std = utils.nanstd(variables_launcher_running['result_em_fits'], axis=-1)

        plt.ion()

        # Memory curve kappa
        def sigmaoutput_plot_kappa(sigmaoutput_space, result_em_fits_mean, result_em_fits_std=None, exp_name='', ax=None):

            if ax is not None:
                plt.figure(ax.get_figure().number)
                ax.hold(False)

            ax = utils.plot_mean_std_area(sigmaoutput_space, result_em_fits_mean[..., 0], result_em_fits_std[..., 0], xlabel='sigma output', ylabel='Memory fidelity', linewidth=3, fmt='o-', markersize=8, label='Noise output effect', ax_handle=ax)

            ax.hold(True)

            ax.set_title("{{exp_name}} {T} {M} {ratio_conj:.2f} {sigmax:.3f} {sigmay:.2f}".format(**variables_launcher_running['all_parameters']).format(exp_name=exp_name))
            ax.legend()
            # ax.set_xlim([0.9, T_space_exp.max()+0.1])
            # ax.set_xticks(range(1, T_space_exp.max()+1))
            # ax.set_xticklabels(range(1, T_space_exp.max()+1))

            ax.get_figure().canvas.draw()

            dataio.save_current_figure('noiseoutput_kappa_%s_T{T}_M{M}_ratio{ratio_conj}_sigmax{sigmax}_sigmay{sigmay}_{{label}}_{{unique_id}}.pdf'.format(**variables_launcher_running['all_parameters']) % (exp_name))

            return ax

        # Plot EM Mixtures proportions
        def sigmaoutput_plot_mixtures(sigmaoutput_space, result_em_fits_mean, result_em_fits_std, exp_name='', ax=None):

            if ax is None:
                _, ax = plt.subplots()

            if ax is not None:
                plt.figure(ax.get_figure().number)
                ax.hold(False)

            # mixture probabilities
            print result_em_fits_mean[..., 1]

            result_em_fits_mean[np.isnan(result_em_fits_mean)] = 0.0
            result_em_fits_std[np.isnan(result_em_fits_std)] = 0.0

            utils.plot_mean_std_area(sigmaoutput_space, result_em_fits_mean[..., 1], result_em_fits_std[..., 1], xlabel='sigma output', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Target')
            ax.hold(True)
            utils.plot_mean_std_area(sigmaoutput_space, result_em_fits_mean[..., 2], result_em_fits_std[..., 2], xlabel='sigma output', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Nontarget')
            utils.plot_mean_std_area(sigmaoutput_space, result_em_fits_mean[..., 3], result_em_fits_std[..., 3], xlabel='sigma output', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Random')

            ax.legend(prop={'size':15})

            ax.set_title("{{exp_name}} {T} {M} {ratio_conj:.2f} {sigmax:.3f} {sigmay:.2f}".format(**variables_launcher_running['all_parameters']).format(exp_name=exp_name))

            ax.set_ylim([0.0, 1.1])
            ax.get_figure().canvas.draw()

            dataio.save_current_figure('memorycurves_emfits_%s_T{T}_M{M}_ratio{ratio_conj}_sigmax{sigmax}_sigmay{sigmay}_{{label}}_{{unique_id}}.pdf'.format(**variables_launcher_running['all_parameters']) % (exp_name))

            return ax

        # Do plots
        plotting_parameters['axes']['ax_sigmaoutput_kappa'] = sigmaoutput_plot_kappa(sigmaoutput_space, result_em_fits_mean, result_em_fits_std, exp_name='kappa', ax=plotting_parameters['axes']['ax_sigmaoutput_kappa'])
        plotting_parameters['axes']['ax_sigmaoutput_mixtures'] = sigmaoutput_plot_mixtures(sigmaoutput_space, result_em_fits_mean, result_em_fits_std, exp_name='mixt probs', ax=plotting_parameters['axes']['ax_sigmaoutput_mixtures'])


    if all_parameters.get('do_plots_during_simulation', False):
        # Define the callback function.
        all_parameters['plots_during_simulation_callback'] = dict(function=do_sigma_output_plot, parameters=plotting_parameters)
        # Run the launcher_do_noise_output_effect, plots are done during the runs automatically
        other_launcher_results = launcher_do_noise_output_effect(all_parameters)
    else:
        # Run the launcher_do_noise_output_effect, will do plots later
        other_launcher_results = launcher_do_noise_output_effect(args)

        # Do the plots
        do_sigma_output_plot(other_launcher_results, plotting_parameters)

    # Return the output of the other launcher.
    return other_launcher_results


def launcher_do_noise_output_effect_withplots_live(args):
    '''
        Just like launcher_do_noise_output_effect_withplots do plots as well, while the simulation is going on
    '''

    all_parameters = utils.argparse_2_dict(args)

    all_parameters['do_plots_during_simulation'] = True

    return launcher_do_noise_output_effect_withplots(all_parameters)



