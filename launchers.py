#!/usr/bin/env python
# encoding: utf-8
"""
launchers.py


Created by Loic Matthey on 2012-10-10
Copyright (c) 2012 . All rights reserved.
"""

import matplotlib.pyplot as plt

from datagenerator import *
from hierarchicalrandomnetwork import *
from randomfactorialnetwork import *
from statisticsmeasurer import *
from slicesampler import *
from utils import *
from dataio import *
from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *


def launcher_do_simple_run(args):
    ''' 
        Basic use-case when playing around with the components.

        Instantiate a simple network and sampler

            inference_method:
                - sample
                - max_lik
    '''
    
    print "Simple run"
    

    # Build the random network
    # sigma_y = 0.02
    # sigma_y = 0.2
    # sigma_x = 0.1
    time_weights_parameters = dict(weighting_alpha=args.alpha, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')
    cued_feature_time = args.T-1


    # 'conj', 'feat', 'mixed'
    if args.code_type == 'conj':

        random_network = RandomFactorialNetwork.create_full_conjunctive(args.M, R=args.R, rcscale=args.rc_scale, rcscale_autoset=args.rcscale_auto)
    elif args.code_type == 'feat':
        random_network = RandomFactorialNetwork.create_full_features(args.M, R=args.R, scale=args.rc_scale, ratio=args.feat_ratio)
    elif args.code_type == 'mixed':
        conj_params = dict(scale_moments=[args.rc_scale, 0.0001], ratio_moments=[1.0, 0.0001])
        feat_params = dict(scale=args.rc_scale2, ratio=args.feat_ratio)
        random_network = RandomFactorialNetwork.create_mixed(args.M, R=args.R, ratio_feature_conjunctive=args.ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
    elif args.code_type == 'wavelet':
        random_network = RandomFactorialNetwork.create_wavelet(args.M, R=args.R, scales_number=5)
    elif args.code_type == 'hierarchical':
        random_network = HierarchialRandomNetwork(args.M, sigma_weights=0.5, sparsity_weights=0.5, M_layer_one=args.M_layer_one, rcscale_layer_one='optimal')
        # random_network = HierarchialRandomNetwork(args.M, sigma_weights=2.0, sparsity_weights=1.0, normalise_weights=True, type_layer_one='feature', rcscale_layer_one=0.1, ratio_layer_one=200., M_layer_one=args.M_layer_one)
    else:
        raise ValueError('Code_type is wrong!')
    
    # Construct the real dataset
    print "Building the database"
    data_gen = DataGeneratorRFN(args.N, args.T, random_network, sigma_y=args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time, stimuli_generation=args.stimuli_generation)
    
    # Measure the noise structure
    print "Measuring noise structure"
    data_gen_noise = DataGeneratorRFN(3000, args.T, random_network, sigma_y=args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time, stimuli_generation=args.stimuli_generation)
    stat_meas = StatisticsMeasurer(data_gen_noise)
    # stat_meas = StatisticsMeasurer(data_gen)
    
    print "Building sampler..."
    sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=cued_feature_time)
    
    print "Inferring optimal angles, for t=%d" % sampler.tc[0]
    # sampler.set_theta_max_likelihood(num_points=500, post_optimise=True)
    
    if args.inference_method == 'sample':
        # Sample thetas
        print "-> Sampling theta"
        sampler.sample_theta(num_samples=args.num_samples, burn_samples=20, selection_method='median', selection_num_samples=args.num_samples, integrate_tc_out=False, debug=False)
    elif args.inference_method == 'max_lik':
        # Just use the ML value for the theta
        print "-> Setting theta to ML values"
        sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
    elif args.inference_method == 'none':
        # Do nothing
        print "do nothing"
        
    sampler.print_comparison_inferred_groundtruth()
    
    return locals()
    


def launcher_do_save_responses_simultaneous(args):
    '''
        Simulate simultaneous presentations, with 1...T objects.
        Outputs the responses and target/non-targets, to be fitted in Matlab (TODO: convert EM fit in Python code)
    '''

    output_dir = os.path.join(args.output_directory, args.label)
    output_string = unique_filename(prefix=strcat(output_dir, 'save_responses_simultaneous'))

    # Build the random network
    time_weights_parameters = dict(weighting_alpha=args.alpha, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')

    print "Doing do_save_responses_simultaneous"
    print "max_T: %s" % args.T
    print "File: %s" % output_string

    all_precisions = np.zeros((args.T, args.T, args.num_repetitions))
    all_responses = np.zeros((args.T, args.T, args.num_repetitions, args.N))
    all_targets = np.zeros((args.T, args.T, args.num_repetitions, args.N))
    all_nontargets = np.zeros((args.T, args.T, args.num_repetitions, args.N, args.T-1))
    
    for repet_i in np.arange(args.num_repetitions):
        
        # Construct different datasets, with t objects
        for t in np.arange(args.T):
            
            if args.code_type == 'conj':
                random_network = RandomFactorialNetwork.create_full_conjunctive(args.M, R=args.R, scale_moments=(args.rc_scale, 0.0001), ratio_moments=(1.0, 0.001))
            elif args.code_type == 'feat':
                random_network = RandomFactorialNetwork.create_full_features(args.M, R=args.R, scale=args.rc_scale, ratio=40.)
            elif args.code_type == 'mixed':
                conj_params = dict(scale_moments=(args.rc_scale, 0.001), ratio_moments=(1.0, 0.0001))
                feat_params = dict(scale=args.rc_scale2, ratio=40.)

                random_network = RandomFactorialNetwork.create_mixed(args.M, R=args.R, ratio_feature_conjunctive=args.ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
            else:
                raise ValueError('Code_type is wrong!')

            
            # Construct the real dataset
            data_gen = DataGeneratorRFN(args.N, t+1, random_network, sigma_y=args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters)
            
            # Measure the noise structure
            data_gen_noise = DataGeneratorRFN(3000, t+1, random_network, sigma_y=args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters)
            stat_meas = StatisticsMeasurer(data_gen_noise)
            # stat_meas = StatisticsMeasurer(data_gen)
            
            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters)
            
            for tc in np.arange(t+1):
                print "Doing T=%d, Tc=%d,  %d/%d" % (t+1, tc, repet_i+1, args.num_repetitions)

                # Change the cued feature
                sampler.change_cued_features(tc)

                # Sample the new theta
                sampler.sample_theta(num_samples=args.num_samples, selection_num_samples=args.selection_num_samples, integrate_tc_out=False, selection_method='median')

                # Save the precision
                # all_precisions[param1_i, repet_i] = sampler.get_precision()
                all_precisions[t, tc, repet_i] = sampler.compute_angle_error()[1]

                print "-> %.5f" % all_precisions[t, tc, repet_i]

                # Save the responses, targets and nontargets
                print t
                print tc
                (all_responses[t, tc, repet_i], all_targets[t, tc, repet_i], all_nontargets[t, tc, repet_i, :, :t])=sampler.collect_responses()

        
            # Save to disk, unique filename
            np.save(output_string, {'all_precisions': all_precisions, 'all_responses': all_responses, 'all_targets': all_targets, 'all_nontargets': all_nontargets, 'args': args, 'num_repetitions': args.num_repetitions, 'T': args.T, 'output_string': output_string})
            sio.savemat(output_string, {'all_precisions': all_precisions, 'all_responses': all_responses, 'all_targets': all_targets, 'all_nontargets': all_nontargets, 'args': args, 'num_repetitions': args.num_repetitions, 'T': args.T, 'output_string': output_string})

    
    print all_precisions

    f = plt.figure()
    ax = f.add_subplot(111)
    for t in np.arange(args.T):
        t_space_aligned_right = (args.T - np.arange(t+1))[::-1]
        semilogy_mean_std_area(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], np.std(1./all_precisions[t], 1)[:t+1], ax_handle=ax)
    ax.set_xlabel('Recall time')
    ax.set_ylabel('Precision [rad]')
    
    print "Done: %s" % output_string
    return locals()



