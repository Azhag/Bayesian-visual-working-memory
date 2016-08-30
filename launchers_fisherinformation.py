#!/usr/bin/env python
# encoding: utf-8
"""
launchers_fisherinformation.py


Created by Loic Matthey on 2012-10-10
Copyright (c) 2012 . All rights reserved.
"""

import utils
# import matplotlib.pyplot as plt
import numpy as np

import dataio as DataIO
import progress

import launchers


def launcher_check_fisher_fit_1obj_2016(args):
    print "Doing a piece of work for launcher_check_fisher_fit_1obj_2016"

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

    # Result arrays
    result_all_precisions = np.nan*np.empty((all_parameters['num_repetitions']), dtype=float)
    result_FI_rc_curv = np.nan*np.empty((all_parameters['N'], all_parameters['num_repetitions']), dtype=float)
    result_FI_rc_theo = np.nan*np.empty((all_parameters['N'], all_parameters['num_repetitions']), dtype=float)
    result_FI_rc_theocov = np.nan*np.empty((all_parameters['N'], all_parameters['num_repetitions']), dtype=float)
    result_FI_rc_theo_largeN = np.nan*np.empty((all_parameters['num_repetitions']), dtype=float)
    result_marginal_inv_FI = np.nan*np.ones((2, all_parameters['num_repetitions']))
    result_marginal_FI = np.nan*np.ones((2, all_parameters['num_repetitions']))

    result_em_fits = np.nan*np.empty((6, all_parameters['num_repetitions']))

    search_progress = progress.Progress(all_parameters['num_repetitions'])

    for repet_i in xrange(all_parameters['num_repetitions']):
        print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

        print "Fisher Info check, rep %d/%d" % (repet_i+1, all_parameters['num_repetitions'])

        ### WORK WORK WORK work? ###

        # Instantiate
        (_, _, _, sampler) = launchers.init_everything(all_parameters)

        # Sample
        sampler.run_inference(all_parameters)

        # Compute precision
        print "get precision..."
        result_all_precisions[repet_i] = sampler.get_precision()

        # Theoretical Fisher info
        print "theoretical FI"
        result_FI_rc_theo[:, repet_i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=False)
        result_FI_rc_theocov[:, repet_i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=True)
        result_FI_rc_theo_largeN[repet_i] = sampler.estimate_fisher_info_theocov_largen(use_theoretical_cov=True)

        # Fisher Info from curvature
        print "Compute fisher from curvature"
        fi_curv_dict = sampler.estimate_fisher_info_from_posterior_avg(num_points=500, full_stats=True)
        result_FI_rc_curv[:, repet_i] = fi_curv_dict['all']

        # Fit mixture model
        print "fit mixture model..."
        curr_params_fit = sampler.fit_mixture_model(use_all_targets=False)
        curr_params_fit['mixt_nontargets_sum'] = np.sum(curr_params_fit['mixt_nontargets'])
        result_em_fits[..., repet_i] = [curr_params_fit[key] for key in ('kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random', 'train_LL', 'bic')]

        # Compute marginal inverse fisher info
        print "compute marginal inverse fisher info"
        marginal_fi_dict = sampler.estimate_marginal_inverse_fisher_info_montecarlo()
        result_marginal_inv_FI[:, repet_i] = [marginal_fi_dict[key]
            for key in ('inv_FI', 'inv_FI_std')]
        result_marginal_FI[:, repet_i] = [marginal_fi_dict[key]
            for key in ('FI', 'FI_std')]

        ## Run callback function if exists
        if plots_during_simulation_callback:
            print "Doing plots..."
            try:
                # Best super safe, if this fails then the simulation must continue!
                plots_during_simulation_callback['function'](locals(), plots_during_simulation_callback['parameters'])
                print "plots done."
            except Exception:
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
