#!/usr/bin/env python
# encoding: utf-8
"""
launchers_hierarchicalnetwork.py


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
from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *

import launchers


def launcher_do_hierarchical_precision_M_Mlower(args):
    '''
        Compare the evolution of the precision curve as the number of neurons in a hierarchical network increases.
    '''

    all_parameters = vars(args)

    M_space = np.linspace(5, 505, 26)
    M_lower_space = np.arange(5, 31, 2)**2.

    M_space = np.array([10, 25, 100])
    M_lower_space = np.array([49, 100])
    T_space = np.arange(1, all_parameters['T']+1)

    results_precision_M_T = np.zeros((M_space.size, M_lower_space.size, T_space.size, num_repetitions), dtype=float)

    # Show the progress
    search_progress = progress.Progress(M_space.size*M_lower_space.size)

    print M_space
    print M_lower_space

    for m_i, M in enumerate(M_space):
        for m_l_i, M_layer_one in enumerate(M_lower_space):
            # Current parameter values
            all_parameters['M']             = M
            all_parameters['M_layer_one']   = M_layer_one

            ### WORK UNIT
            output = launcher_do_hierarchical_precision_M_Mlower_pbs(all_parameters)

            results_precision_M_T[m_i, m_l_i] = output['results_precision_M_T'][0, 0]
            ### DONE WORK UNIT

            search_progress.increment()

            if run_counter % save_every == 0 or search_progress.done():
                dataio.save_variables(variables_to_save, locals())

                run_counter += 1

    return locals()


def launcher_do_hierarchical_precision_M_sparsity_sigmaweight_feature(args):
    '''
        Compare the evolution of the precision curve as the sparsity, sigma and M change, for a hierarchical code with feature base
    '''

    all_parameters = vars(args)

    code_type = 'hierarchical'

    dataio = DataIO(output_folder=args.output_directory, label=args.label)
    # variables_to_save = ['M_space', 'T_space',  'repet_i', 'num_repetitions', 'results_precision_N', 'all_responses', 'all_targets', 'all_nontargets']
    variables_to_save = ['M_space', 'T_space', 'sparsity_space', 'sigma_weights_space', 'repet_i', 'num_repetitions', 'results_precision_N']

    save_every = 5
    run_counter = 0

    num_repetitions = all_parameters['num_repetitions']

    # M_space = np.array([all_parameters['M']])
    # M_space = np.array([4*4, 5*5, 7*7, 8*8, 9*9, 10*10, 15*15, 20*20])
    M_space = np.linspace(5, 500, 10)
    sparsity_space = np.linspace(0.01, 1.0, 10.)
    sigma_weights_space = np.linspace(0.1, 2.0, 10)
    T_space = np.arange(1, all_parameters['T']+1)

    results_precision_M_T = np.zeros((M_space.size, sparsity_space.size, sigma_weights_space.size, T_space.size, num_repetitions), dtype=float)
    # all_responses = np.zeros((M_space.size, sparsity_space.size, sigma_weights_space.size, T_space.size, num_repetitions, all_parameters['N']))
    # all_targets = np.zeros((M_space.size, sparsity_space.size, sigma_weights_space.size, T_space.size, num_repetitions, all_parameters['N']))
    # all_nontargets = np.zeros((M_space.size, sparsity_space.size, sigma_weights_space.size, T_space.size, num_repetitions, all_parameters['N'], all_parameters['T']-1))

    all_parameters['type_layer_one'] = 'feature'

    # Show the progress
    search_progress = progress.Progress(T_space.size*M_space.size*sigma_weights_space.size*sparsity_space.size*num_repetitions)

    print M_space
    print sparsity_space
    print sigma_weights_space
    print T_space

    for repet_i in xrange(num_repetitions):
        for m_i, M in enumerate(M_space):
            for s_i, sparsity in enumerate(sparsity_space):
                for sw_i, sigma_weights in enumerate(sigma_weights_space):
                    for t_i, t in enumerate(T_space):
                    # Will estimate the precision

                        print "Precision as function of N, hierarchical network, T: %d/%d, M %d, sparsity %.3f, weights: %.2f, (%d/%d). %.2f%%, %s left - %s" % (t, T_space[-1], M, sparsity, sigma_weights, repet_i+1, num_repetitions, search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

                        # Current parameter values
                        all_parameters['M']             = M
                        all_parameters['T']             = t
                        all_parameters['code_type']     = code_type
                        all_parameters['sparsity']      = sparsity
                        all_parameters['sigma_weights'] = sigma_weights

                        ### WORK UNIT
                        (random_network, data_gen, stat_meas, sampler) = launchers.init_everything(all_parameters)

                        if all_parameters['inference_method'] == 'sample':
                            # Sample thetas
                            sampler.sample_theta(num_samples=all_parameters['num_samples'], burn_samples=100, selection_method=all_parameters['selection_method'], selection_num_samples=all_parameters['selection_num_samples'], integrate_tc_out=False, debug=False)
                        elif all_parameters['inference_method'] == 'max_lik':
                            # Just use the ML value for the theta
                            sampler.set_theta_max_likelihood(num_points=150, post_optimise=True)

                        results_precision_M_T[m_i, s_i, sw_i, t_i, repet_i] = sampler.get_precision()
                        print results_precision_M_T[m_i, s_i, sw_i, t_i, repet_i]

                        # (all_responses[m_i, s_i, t_i, repet_i], all_targets[m_i, s_i, t_i, repet_i], all_nontargets[m_i, s_i, t_i, repet_i, :, :t_i]) = sampler.collect_responses()

                        ### DONE WORK UNIT

                        search_progress.increment()

                        if run_counter % save_every == 0 or search_progress.done():
                            dataio.save_variables(variables_to_save, locals())

                        run_counter += 1

    return locals()



###### PBS runners #####

def launcher_do_hierarchical_precision_M_Mlower_pbs(args):
    '''
        Compare the evolution of the precision curve as the number of neurons in a hierarchical network increases.
    '''


    print "Doing a piece of work for launcher_do_hierarchical_precision_M_Mlower_pbs"
    save_all_output = False

    try:
        # Convert Argparse.Namespace to dict
        all_parameters = vars(args)
    except TypeError:
        # Assume it's already done
        assert type(args) is dict, "args is neither Namespace nor dict, WHY?"
        all_parameters = args

    code_type = 'hierarchical'

    dataio = DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'])
    variables_to_save = ['M_space', 'T_space', 'M_lower_space', 'repet_i', 'num_repetitions', 'results_precision_M_T']

    save_every = 5
    run_counter = 0

    num_repetitions = all_parameters['num_repetitions']

    M_space = np.array([all_parameters['M']])
    M_lower_space = np.array([all_parameters['M_layer_one']])
    T_space = np.arange(1, all_parameters['T']+1)

    results_precision_M_T = np.zeros((M_space.size, M_lower_space.size, T_space.size, num_repetitions), dtype=float)

    if save_all_output:
        variables_to_save.extend(['all_responses', 'all_targets', 'all_nontargets'])

        all_responses = np.zeros((M_space.size, M_lower_space.size, T_space.size, num_repetitions, all_parameters['N']))
        all_targets = np.zeros((M_space.size, M_lower_space.size, T_space.size, num_repetitions, all_parameters['N']))
        all_nontargets = np.zeros((M_space.size, M_lower_space.size, T_space.size, num_repetitions, all_parameters['N'], all_parameters['T']-1))

    # Show the progress
    search_progress = progress.Progress(T_space.size*M_space.size*M_lower_space.size*num_repetitions)

    print M_space
    print M_lower_space
    print T_space

    for repet_i in xrange(num_repetitions):
        for m_i, M in enumerate(M_space):
            for m_l_i, M_layer_one in enumerate(M_lower_space):
                for t_i, t in enumerate(T_space):
                    # Will estimate the precision

                    print "Precision as function of N, hierarchical network, T: %d/%d, M %d, M_layer_one %d, (%d/%d). %.2f%%, %s left - %s" % (t, T_space[-1], M, M_layer_one, repet_i+1, num_repetitions, search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

                    # Current parameter values
                    all_parameters['M']             = M
                    all_parameters['T']             = t
                    all_parameters['code_type']     = code_type
                    all_parameters['M_layer_one']   = M_layer_one

                    ### WORK UNIT
                    (random_network, data_gen, stat_meas, sampler) = launchers.init_everything(all_parameters)

                    if all_parameters['inference_method'] == 'sample':
                        # Sample thetas
                        sampler.sample_theta(num_samples=all_parameters['num_samples'], burn_samples=100, selection_method=all_parameters['selection_method'], selection_num_samples=all_parameters['selection_num_samples'], integrate_tc_out=False, debug=False)
                    elif all_parameters['inference_method'] == 'max_lik':
                        # Just use the ML value for the theta
                        sampler.set_theta_max_likelihood(num_points=100, post_optimise=True)

                    results_precision_M_T[m_i, m_l_i, t_i, repet_i] = sampler.get_precision()
                    print results_precision_M_T[m_i, m_l_i, t_i, repet_i]

                    if save_all_output:
                        (all_responses[m_i, m_l_i, t_i, repet_i], all_targets[m_i, m_l_i, t_i, repet_i], all_nontargets[m_i, m_l_i, t_i, repet_i, :, :t_i]) = sampler.collect_responses()

                    ### DONE WORK UNIT

                    search_progress.increment()

                    if run_counter % save_every == 0 or search_progress.done():
                        dataio.save_variables(variables_to_save, locals())

                    run_counter += 1

    print "All finished"

    return locals()


def launcher_do_hierarchical_precision_M_sparsity_sigmaweight_feature_pbs(args):
    '''
        Compare the evolution of the precision curve as the sparsity, sigma and M change, for a hierarchical code with feature base
    '''

    print "Doing a piece of work for launcher_do_hierarchical_precision_M_sparsity_sigmaweight_feature_pbs"
    save_all_output = False


    all_parameters = vars(args)

    code_type = 'hierarchical'

    dataio = DataIO(output_folder=args.output_directory, label=args.label)
    variables_to_save = ['M_space', 'T_space', 'sparsity_space', 'sigma_weights_space', 'repet_i', 'num_repetitions', 'results_precision_M_T']

    save_every = 5
    run_counter = 0

    num_repetitions = all_parameters['num_repetitions']

    # M_space = np.array([all_parameters['M']])
    # M_space = np.array([4*4, 5*5, 7*7, 8*8, 9*9, 10*10, 15*15, 20*20])
    M_space = np.array([all_parameters['M']])
    sparsity_space = np.array([all_parameters['sparsity']])
    sigma_weights_space = np.array([all_parameters['sigma_weights']])
    T_space = np.arange(1, all_parameters['T']+1)


    results_precision_M_T = np.zeros((M_space.size, sparsity_space.size, sigma_weights_space.size, T_space.size, num_repetitions), dtype=float)
    if save_all_output:
        variables_to_save.extend(['all_responses', 'all_targets', 'all_nontargets'])

        all_responses = np.zeros((M_space.size, sparsity_space.size, sigma_weights_space.size, T_space.size, num_repetitions, all_parameters['N']))
        all_targets = np.zeros((M_space.size, sparsity_space.size, sigma_weights_space.size, T_space.size, num_repetitions, all_parameters['N']))
        all_nontargets = np.zeros((M_space.size, sparsity_space.size, sigma_weights_space.size, T_space.size, num_repetitions, all_parameters['N'], all_parameters['T']-1))


    all_parameters['type_layer_one'] = 'feature'

    # Show the progress
    search_progress = progress.Progress(T_space.size*M_space.size*sigma_weights_space.size*sparsity_space.size*num_repetitions)

    print M_space
    print sparsity_space
    print sigma_weights_space
    print T_space

    for repet_i in xrange(num_repetitions):
        for m_i, M in enumerate(M_space):
            for s_i, sparsity in enumerate(sparsity_space):
                for sw_i, sigma_weights in enumerate(sigma_weights_space):
                    for t_i, t in enumerate(T_space):
                    # Will estimate the precision

                        print "Precision as function of N, hierarchical network, T: %d/%d, M %d, sparsity %.3f, weights: %.2f, (%d/%d). %.2f%%, %s left - %s" % (t, T_space[-1], M, sparsity, sigma_weights, repet_i+1, num_repetitions, search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

                        # Current parameter values
                        all_parameters['M']             = M
                        all_parameters['T']             = t
                        all_parameters['code_type']     = code_type
                        all_parameters['sparsity']      = sparsity
                        all_parameters['sigma_weights'] = sigma_weights

                        ### WORK UNIT
                        (random_network, data_gen, stat_meas, sampler) = launchers.init_everything(all_parameters)

                        if all_parameters['inference_method'] == 'sample':
                            # Sample thetas
                            sampler.sample_theta(num_samples=all_parameters['num_samples'], burn_samples=100, selection_method=all_parameters['selection_method'], selection_num_samples=all_parameters['selection_num_samples'], integrate_tc_out=False, debug=False)
                        elif all_parameters['inference_method'] == 'max_lik':
                            # Just use the ML value for the theta
                            sampler.set_theta_max_likelihood(num_points=150, post_optimise=True)

                        results_precision_M_T[m_i, s_i, sw_i, t_i, repet_i] = sampler.get_precision()
                        print results_precision_M_T[m_i, s_i, sw_i, t_i, repet_i]

                        if save_all_output:
                            (all_responses[m_i, m_l_i, t_i, repet_i], all_targets[m_i, m_l_i, t_i, repet_i], all_nontargets[m_i, m_l_i, t_i, repet_i, :, :t_i]) = sampler.collect_responses()

                        ### DONE WORK UNIT

                        search_progress.increment()

                        if run_counter % save_every == 0 or search_progress.done():
                            dataio.save_variables(variables_to_save, locals())

                        run_counter += 1

    print "All finished"

    return locals()







