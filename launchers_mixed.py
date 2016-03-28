#!/usr/bin/env python
# encoding: utf-8
"""
launchers_mixed.py


Created by Loic Matthey on 2013-05-18
Copyright (c) 2013 . All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np

from randomfactorialnetwork import *
from datagenerator import *
from statisticsmeasurer import *
from slicesampler import *
from utils import *
from dataio import *
# from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *
import em_circularmixture

import launchers


def launcher_do_mixed_varyratio_precision_pbs(args):
    '''
        Compare the evolution of the precision curve as the number of neurons in a mixed network changes.
    '''


    print "Doing a piece of work for launcher_do_mixed_varyratio_precision_pbs"
    save_all_output = False

    try:
        # Convert Argparse.Namespace to dict
        all_parameters = vars(args)
    except TypeError:
        # Assume it's already done
        assert type(args) is dict, "args is neither Namespace nor dict, WHY?"
        all_parameters = args

    code_type = 'mixed'

    dataio = DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))

    save_every = 5
    run_counter = 0

    num_repetitions = all_parameters['num_repetitions']

    ratio_space = np.array([all_parameters['ratio_conj']])
    T_space = np.arange(1, all_parameters['T']+1)

    results_precision_ratio_T = np.nan*np.empty((ratio_space.size, T_space.size, num_repetitions), dtype=float)

    # if save_all_output:
    #     results_all_responses = np.nan*np.empty((M_space.size, M_lower_space.size, T_space.size, num_repetitions, all_parameters['N']))
    #     results_all_targets = np.nan*np.empty((M_space.size, M_lower_space.size, T_space.size, num_repetitions, all_parameters['N']))
    #     results_all_nontargets = np.nan*np.empty((M_space.size, M_lower_space.size, T_space.size, num_repetitions, all_parameters['N'], all_parameters['T']-1))

    # Show the progress
    search_progress = progress.Progress(T_space.size*ratio_space.size*num_repetitions)

    print T_space
    print ratio_space

    for repet_i in xrange(num_repetitions):
        for ratio_conj_i, ratio_conj in enumerate(ratio_space):
            for t_i, t in enumerate(T_space):
                # Will estimate the precision

                print "Precision as function of N, hierarchical network, T: %d/%d, ratio_conj %.2f, (%d/%d). %.2f%%, %s left - %s" % (t, T_space[-1], ratio_conj, repet_i+1, num_repetitions, search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

                # Current parameter values
                all_parameters['T']             = t
                all_parameters['code_type']     = code_type
                all_parameters['ratio_conj']    = ratio_conj

                ### WORK UNIT
                (random_network, data_gen, stat_meas, sampler) = launchers.init_everything(all_parameters)

                if all_parameters['inference_method'] == 'sample':
                    # Sample thetas
                    sampler.sample_theta(num_samples=all_parameters['num_samples'], burn_samples=100, selection_method=all_parameters['selection_method'], selection_num_samples=all_parameters['selection_num_samples'], integrate_tc_out=False, debug=False)
                elif all_parameters['inference_method'] == 'max_lik':
                    # Just use the ML value for the theta
                    sampler.set_theta_max_likelihood(num_points=100, post_optimise=True)

                results_precision_ratio_T[ratio_conj_i, t_i, repet_i] = sampler.get_precision()
                print results_precision_ratio_T[ratio_conj_i, t_i, repet_i]

                # if save_all_output:
                #     (results_all_responses[ratio_conj_i, t_i, repet_i], results_all_targets[ratio_conj_i, t_i, repet_i], results_all_nontargets[ratio_conj_i, t_i, repet_i, :, :t_i]) = sampler.collect_responses()

                ### DONE WORK UNIT

                search_progress.increment()

                if run_counter % save_every == 0 or search_progress.done():
                    dataio.save_variables_default(locals())

                run_counter += 1

    print "All finished"

    return locals()

