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
    if all_parameters['subaction'] == 'collect_responses':
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
            curr_params_fit = sampler.fit_mixture_model(use_all_targets=True)
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

    ### Run the launcher_do_memory_curve_marginal_fi, will do plots later
    other_launcher_results = launcher_do_memory_curve_marginal_fi(args)

    if other_launcher_results['all_parameters']['ratio_hierarchical'] is not None:
        # Use ratio_conj for plotting/titles
        other_launcher_results['all_parameters']['ratio_conj'] = other_launcher_results['all_parameters']['ratio_hierarchical']

    dataio = other_launcher_results['dataio']
    T_space = other_launcher_results['T_space']

    result_em_fits_mean = np.mean(other_launcher_results['result_em_fits'], axis=-1)
    result_em_fits_std = np.std(other_launcher_results['result_em_fits'], axis=-1)

    ### Load the experimental data for the plots
    data_gorgo11 = load_experimental_data.load_data_gorgo11(fit_mixture_model=True)
    gorgo11_T_space = data_gorgo11['data_to_fit']['n_items']
    gorgo11_emfits_meanstd = data_gorgo11['em_fits_nitems_arrays']
    gorgo11_emfits_meanstd['mean'][2, 0] = 0.0
    gorgo11_emfits_meanstd['std'][2, 0] = 0.0

    data_bays09 = load_experimental_data.load_data_bays09(fit_mixture_model=True)
    bays09_T_space = data_bays09['data_to_fit']['n_items']
    bays09_emfits_meanstd = data_bays09['em_fits_nitems_arrays']

    dict_experiments_to_plot = dict(bays09=dict(emfits=bays09_emfits_meanstd, T_space=bays09_T_space), gorgo11=dict(emfits=gorgo11_emfits_meanstd, T_space=gorgo11_T_space))

    ### Now do the plots
    # Memory curve kappa
    def mem_plot_kappa(T_space_exp, exp_kappa_mean, exp_kappa_std=None, exp_name=''):
        ax = utils.plot_mean_std_area(T_space_exp, exp_kappa_mean, exp_kappa_std, linewidth=3, fmt='o-', markersize=8, label='Experimental data')

        ax = utils.plot_mean_std_area(T_space[:T_space_exp.max()], result_em_fits_mean[..., :T_space_exp.max(), 0], result_em_fits_std[..., :T_space_exp.max(), 0], xlabel='Number of items', ylabel="Memory error $[rad^{-2}]$", linewidth=3, fmt='o-', markersize=8, label='Fitted kappa', ax_handle=ax)

        ax.set_title("{{exp_name}} {M} {ratio_conj} {sigmax} {sigmay}".format(**other_launcher_results['all_parameters']).format(exp_name=exp_name))
        ax.legend()
        ax.set_xlim([0.9, T_space_exp.max()+0.1])
        ax.set_xticks(range(1, T_space_exp.max()+1))
        ax.set_xticklabels(range(1, T_space_exp.max()+1))

        ax.get_figure().canvas.draw()

        dataio.save_current_figure('memorycurves_kappa_%s_M{M}_ratio{ratio_conj}_sigmax{sigmax}_sigmay{sigmay}_{{label}}_{{unique_id}}.pdf'.format(**other_launcher_results['all_parameters']) % (exp_name))

    # Plot EM Mixtures proportions
    def em_plot_paper(exp_name=''):
        f, ax = plt.subplots()

        # mixture probabilities
        utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 1], result_em_fits_std[..., 1], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Target')
        utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 2], result_em_fits_std[..., 2], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Nontarget')
        utils.plot_mean_std_area(T_space, result_em_fits_mean[..., 3], result_em_fits_std[..., 3], xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Random')

        ax.legend(prop={'size':15})

        ax.set_title("{{exp_name}} {M} {ratio_conj} {sigmax} {sigmay}".format(**other_launcher_results['all_parameters']).format(exp_name=exp_name))
        ax.set_xlim([1.0, T_space.size])
        ax.set_ylim([0.0, 1.1])
        ax.set_xticks(range(1, T_space.size+1))
        ax.set_xticklabels(range(1, T_space.size+1))

        f.canvas.draw()

        dataio.save_current_figure('memorycurves_emfits_%s_M{M}_ratio{ratio_conj}_sigmax{sigmax}_sigmay{sigmay}_{{label}}_{{unique_id}}.pdf'.format(**other_launcher_results['all_parameters']) % (exp_name))

    # Do all plots for all datasets
    for exper_name, exper_data in dict_experiments_to_plot.iteritems():
        mem_plot_kappa(exper_data['T_space'], exper_data['emfits']['mean'][0], exper_data['emfits']['std'][0], exp_name=exper_name)
        em_plot_paper(exp_name=exper_name)



    # Return the output of the other launcher.
    return other_launcher_results



