#!/usr/bin/env python
# encoding: utf-8
"""
launchers_fitexperiment.py


Created by Loic Matthey on 2013-09-30
Copyright (c) 2013 . All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np

import launchers

from utils import *
from dataio import *
from fitexperiment import *


def launcher_do_fitexperiment(args):
    '''
        Perform a simple estimation of the loglikelihood of the data, under a model with provided parameters
    '''

    print "Doing a piece of work for launcher_do_fitexperiment"

    try:
        # Convert Argparse.Namespace to dict
        all_parameters = vars(args)
    except TypeError:
        # Assume it's already done
        assert type(args) is dict, "args is neither Namespace nor dict, WHY?"
        all_parameters = args

    print all_parameters

    # Force some parameters
    # all_parameters['experiment_id'] = "dualrecall"
    all_parameters['experiment_id'] = "gorgo_simult"
    all_parameters['experiment_params'] = dict(n_items_to_fit=all_parameters['T'])

    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))

    # Result arrays
    result_log_posterior_mean = np.nan*np.ones(1)
    result_log_posterior_std = np.nan*np.ones(1)

    ### WORK WORK WORK work? ###

    fit_exp = FitExperiment(all_parameters)
    result_log_posterior_mean, result_log_posterior_std = fit_exp.fit_parameter_pbs()

    ### /Work ###
    dataio.save_variables_default(locals())

    #### Plots ###

    print "All finished"

    return locals()



def launcher_do_fitexperiment_mixed_tworcscale(args):
    '''
        Estimate of the loglikelihood of the data, under a model with provided parameters
        Vary rcscale and rcscale2, to use with a mixed network

    '''

    print "Doing a piece of work for launcher_do_fitexperiment_mixed_tworcscale"

    try:
        # Convert Argparse.Namespace to dict
        all_parameters = vars(args)
    except TypeError:
        # Assume it's already done
        assert type(args) is dict, "args is neither Namespace nor dict, WHY?"
        all_parameters = args

    print all_parameters

    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))
    save_every = 1
    run_counter = 0

    # Initialize FitExperiment
    fit_exp = FitExperiment(all_parameters)

    # Parameters to vary
    param1_space = np.linspace(0.01, 10, 15)    # kappa conj
    param2_space = np.linspace(0.01, 30., 17)   # kappa feat

    # Result arrays
    result_log_posterior_mean = np.nan*np.ones((param1_space.size, param2_space.size))
    result_log_posterior_std = np.nan*np.ones((param1_space.size, param2_space.size))

    search_progress = progress.Progress(param1_space.size*param2_space.size)

    for i, rc_scale in enumerate(param1_space):
        for j, rc_scale2 in enumerate(param2_space):
            print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

            print "Fit for rc_scale=%.2f, rc_scale2=%.2f" % (rc_scale, rc_scale2)

            # Update parameter
            fit_exp.parameters['rc_scale'] = rc_scale
            fit_exp.parameters['rc_scale2'] = rc_scale2

            ### WORK WORK WORK work? ###

            # Compute the loglikelihood
            result_log_posterior_mean[i, j], result_log_posterior_std[i, j] = fit_exp.estimate_likelihood_multiple_models(num_models=all_parameters['num_repetitions'])

            ### /Work ###

            search_progress.increment()
            if run_counter % save_every == 0 or search_progress.done():
                dataio.save_variables_default(locals())
            run_counter += 1

    # Finished
    dataio.save_variables_default(locals())

    # Plot the result
    # plot_mean_std_area(M_space, self.llh_fullspace_mean, self.llh_fullspace_std)
    # dataio.save_variables(variables_to_save, locals())
    pcolor_2d_data(result_log_posterior_mean, param1_space, param2_space, "rc_scale", "rc_scale2")
    dataio.save_current_figure("fitexperiment_mixed_tworcscale_ratio{ratio_conj}_sigmax{sigmax}".format(**all_parameters) + "_{unique_id}.pdf")

    print "All finished"
    return locals()


