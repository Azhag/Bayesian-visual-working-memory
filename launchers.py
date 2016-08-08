#!/usr/bin/env python
# encoding: utf-8
"""
launchers.py


Created by Loic Matthey on 2012-10-10
Copyright (c) 2012 . All rights reserved.
"""

from datagenerator import *
from hierarchicalrandomnetwork import *
from randomfactorialnetwork import *
from statisticsmeasurer import *
from slicesampler import *
from utils import *
from dataio import *
# from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *
from sampler_invtransf_randomfactorialnetwork import *

from highdimensionnetwork import *



################### INITIALISERS ####################
# Define everything here, so that other launchers can call them directly (unless they do something funky)

def init_everything(parameters):

    # Forces some parameters
    parameters['time_weights_parameters'] = dict(weighting_alpha=parameters['alpha'], weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')

    if parameters.get('fixed_cued_feature_time', -1) >= 0:
        parameters['cued_feature_time'] = parameters['fixed_cued_feature_time']
    else:
        parameters['cued_feature_time'] = parameters['T'] - 1

    # Build the random network
    random_network = init_random_network(parameters)

    # print "Building the database"
    data_gen = init_data_gen(random_network, parameters)

    # Measure the noise structure
    stat_meas = init_stat_measurer(random_network, parameters)

    # Init sampler
    sampler = Sampler(data_gen, n_parameters=stat_meas.model_parameters, tc=parameters['cued_feature_time'], sigma_output=parameters['sigma_output'], parameters_dict=parameters, renormalize_sigma_output=parameters.get('renormalize_sigma_output', False), lapse_rate=parameters['lapse_rate'])

    return (random_network, data_gen, stat_meas, sampler)



def init_random_network(parameters):

    # Build the random network

    if parameters['code_type'] == 'conj':
        random_network = HighDimensionNetwork.create_full_conjunctive(parameters['M'], R=parameters['R'], rcscale=parameters['rc_scale'], autoset_parameters=parameters['autoset_parameters'], response_maxout=parameters['response_maxout'])
    elif parameters['code_type'] == 'feat':
        random_network = HighDimensionNetwork.create_full_features(parameters['M'], R=parameters['R'], scale=parameters['rc_scale'], ratio=parameters['feat_ratio'], autoset_parameters=parameters['autoset_parameters'], response_maxout=parameters['response_maxout'])
    elif parameters['code_type'] == 'mixed':
        conj_params = dict(scale=parameters['rc_scale'])
        feat_params = dict(scale=parameters['rc_scale2'], ratio=parameters['feat_ratio'])

        random_network = HighDimensionNetwork.create_mixed(parameters['M'], R=parameters['R'], ratio_feature_conjunctive=parameters['ratio_conj'], conjunctive_parameters=conj_params, feature_parameters=feat_params, autoset_parameters=parameters['autoset_parameters'], response_maxout=parameters['response_maxout'])
    elif parameters['code_type'] == 'wavelet':
        random_network = RandomFactorialNetwork.create_wavelet(parameters['M'], R=parameters['R'], scales_number=5)
    elif parameters['code_type'] == 'hierarchical':
        random_network = HierarchialRandomNetwork(parameters['M'], M_layer_one=parameters['M_layer_one'], optimal_coverage=True, sparsity_weights=parameters['sparsity'], normalise_weights=parameters['normalise_weights'], sigma_weights=parameters['sigma_weights'], type_layer_one=parameters['type_layer_one'], distribution_weights=parameters['distribution_weights'], threshold=parameters['threshold'], output_both_layers=parameters['output_both_layers'], ratio_hierarchical=parameters['ratio_hierarchical'])
    else:
        raise ValueError('Code_type is wrong!')

    return random_network


def init_data_gen(random_network, parameters):
    '''
        Initialisating the DataGenerator
    '''

    return DataGeneratorRFN(parameters['N'], parameters['T'], random_network, sigma_x=parameters['sigmax'], sigma_y=parameters['sigmay'], sigma_baseline=parameters['sigma_baseline'], renormalize_sigma=parameters.get('renormalize_sigma', False), time_weights_parameters=parameters['time_weights_parameters'], cued_feature_time=parameters['cued_feature_time'], stimuli_generation=parameters.get('stimuli_generation', None), enforce_first_stimulus=parameters['enforce_first_stimulus'], stimuli_to_use=parameters.get('stimuli_to_use', None), enforce_min_distance=parameters.get('enforce_min_distance', 0.0), specific_stimuli_random_centers=parameters.get('specific_stimuli_random_centers', True), specific_stimuli_asymmetric=parameters.get('specific_stimuli_asymmetric', False), enforce_distance_cued_feature_only=parameters.get('enforce_distance_cued_feature_only', False), debug=True)


def init_stat_measurer(random_network, parameters):
    # print "Measuring noise structure"
    data_gen_noise = DataGeneratorRFN(5000, parameters['T'], random_network, sigma_y=parameters['sigmay'], sigma_x=parameters['sigmax'], sigma_baseline=parameters['sigma_baseline'], time_weights_parameters=parameters['time_weights_parameters'], cued_feature_time=parameters['cued_feature_time'], stimuli_generation=parameters['stimuli_generation_recall'], enforce_min_distance=parameters.get('enforce_min_distance', 0.0), enforce_distance_cued_feature_only=parameters.get('enforce_distance_cued_feature_only', False), renormalize_sigma=parameters.get('renormalize_sigma', False))
    stat_meas = StatisticsMeasurer(data_gen_noise)

    return stat_meas


def launcher_do_simple_run(args):
    '''
        Basic use-case when playing around with the components.

        Instantiate a simple network and sampler

            inference_method:
                - sample
                - max_lik
    '''

    print "Simple run"

    all_parameters = vars(args)

    (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)

    print "Inferring optimal angles, for t=%d" % sampler.tc[0]
    # sampler.set_theta_max_likelihood(num_points=500, post_optimise=True)

    sampler.run_inference()

    sampler.print_comparison_inferred_groundtruth()

    return locals()





