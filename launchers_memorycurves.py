#!/usr/bin/env python
# encoding: utf-8
"""
launchers_memorycurves.py


Created by Loic Matthey on 2012-10-10
Copyright (c) 2012 . All rights reserved.
"""

import matplotlib.pyplot as plt

from datagenerator import *
from randomfactorialnetwork import *
from statisticsmeasurer import *
from slicesampler import *
from utils import *
from dataio import *
from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *


def launcher_do_memory_curve(args):
    '''
        Get the memory curve
    '''

    
    output_dir = os.path.join(args.output_directory, args.label)
    output_string = unique_filename(prefix=strcat(output_dir, 'memory_curve'))

    # Build the random network
    time_weights_parameters = dict(weighting_alpha=args.alpha, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')

    print "Doing do_memory_curve"
    print "max_T: %s" % args.T
    print "File: %s" % output_string

    all_precisions = np.ones((args.T, args.num_repetitions))*np.nan
    for repet_i in np.arange(args.num_repetitions):
        #### Get multiple examples of precisions, for different number of neurons. #####
        
        if args.code_type == 'conj':
            random_network = RandomFactorialNetwork.create_full_conjunctive(args.M, R=args.R, scale_moments=(args.rc_scale, 0.0001), ratio_moments=(1.0, 0.0001))
        elif args.code_type == 'feat':
            random_network = RandomFactorialNetwork.create_full_features(args.M, R=args.R, scale=args.rc_scale, ratio=40.)
        elif args.code_type == 'mixed':
            conj_params = dict(scale_moments=(args.rc_scale, 0.001), ratio_moments=(1.0, 0.0001))
            feat_params = dict(scale=args.rc_scale2, ratio=40.)
            random_network = RandomFactorialNetwork.create_mixed(args.M, R=args.R, ratio_feature_conjunctive=args.ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
        else:
            raise ValueError('Code_type is wrong!')

        
        # Construct the real dataset
        data_gen = DataGeneratorRFN(args.N, args.T, random_network, sigma_y=args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters)
        
        # Measure the noise structure
        data_gen_noise = DataGeneratorRFN(3000, args.T, random_network, sigma_y=args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters)
        stat_meas = StatisticsMeasurer(data_gen_noise)
        # stat_meas = StatisticsMeasurer(data_gen)
        
        sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters)
        
        for t in np.arange(args.T):
            print "Doing T=%d,  %d/%d" % (t, repet_i+1, args.num_repetitions)

            # Change the cued feature
            sampler.change_cued_features(t)

            # Cheat here, just use the ML value for the theta
            # sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
            # sampler.set_theta_max_likelihood_tc_integratedout(num_points=200, post_optimise=True)

            # Sample the new theta
            sampler.sample_theta(num_samples=args.num_samples, burn_samples=20, selection_method='median', selection_num_samples=args.selection_num_samples, integrate_tc_out=False, debug=False)
            
            # Save the precision
            all_precisions[t, repet_i] = sampler.get_precision(remove_chance_level=True)
            # all_precisions[t, repet_i] = sampler.compute_angle_error()[1]

            print "-> %.5f" % all_precisions[t, repet_i]
    
        # Save to disk, unique filename
        np.save(output_string, {'all_precisions': all_precisions, 'args': args, 'num_repetitions': args.num_repetitions, 'T': args.T, 'output_string': output_string})

    print all_precisions

    f = plt.figure()
    ax = f.add_subplot(111)
    plot_mean_std_area(np.arange(args.T), np.mean(all_precisions, 1), np.std(all_precisions, 1), ax_handle=ax)
    ax.set_xlabel('Object')
    ax.set_ylabel('Precision [rad]')
    
    print "Done: %s" % output_string
    return locals()


def launcher_do_multiple_memory_curve(args):
    '''
        Get the memory curves, for 1...T objects
    '''

    output_dir = os.path.join(args.output_directory, args.label)
    output_string = unique_filename(prefix=strcat(output_dir, 'multiple_memory_curve'))

    # Build the random network
    time_weights_parameters = dict(weighting_alpha=args.alpha, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')

    print "Doing do_multiple_memory_curve"
    print "max_T: %s" % args.T
    print "File: %s" % output_string

    all_precisions = np.zeros((args.T, args.T, args.num_repetitions))

    # Construct different datasets, with t objects
    for t in np.arange(args.T):

        for repet_i in np.arange(args.num_repetitions):
            #### Get multiple examples of precisions, for different number of neurons. #####
            
            if args.code_type == 'conj':
                random_network = RandomFactorialNetwork.create_full_conjunctive(args.M, R=args.R, scale_moments=(args.rc_scale, 0.0001), ratio_moments=(1.0, 0.0001))
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

                # Cheat here, just use the ML value for the theta
                # sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
                # sampler.set_theta_max_likelihood_tc_integratedout(num_points=200, post_optimise=True)

                # Sample thetas
                sampler.sample_theta(num_samples=args.num_samples, burn_samples=20, selection_method='median', selection_num_samples=args.num_samples, integrate_tc_out=False, debug=False)

                # Save the precision
                all_precisions[t, tc, repet_i] = sampler.get_precision(remove_chance_level=True)
                # all_precisions[t, tc, repet_i] = sampler.compute_angle_error()[1]

                print "-> %.5f" % all_precisions[t, tc, repet_i]
            
            # Save to disk, unique filename
            np.save(output_string, {'all_precisions': all_precisions, 'args': args, 'num_repetitions': args.num_repetitions, 'output_string': output_string})

    print all_precisions

    f = plt.figure()
    ax = f.add_subplot(111)
    for t in np.arange(args.T):
        t_space_aligned_right = (args.T - np.arange(t+1))[::-1]
        plot_mean_std_area(t_space_aligned_right, np.mean(all_precisions[t], 1)[:t+1], np.std(all_precisions[t], 1)[:t+1], ax_handle=ax)
    ax.set_xlabel('Recall time')
    ax.set_ylabel('Precision [rad]')
    
    print "Done: %s" % output_string
    return locals()


def launcher_do_multiple_memory_curve_simult(args):
    '''
        Get the memory curves, for 1...T objects, simultaneous presentation
        (will force alpha=1, and only recall one object for each T, more independent)
    '''

    # Should collect all responses?
    collect_responses = True

    # Build the random network
    alpha = 1.
    time_weights_parameters = dict(weighting_alpha=alpha, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')

    # Initialise the output file
    dataio = DataIO(output_folder=args.output_directory, label=args.label)
    output_string = dataio.filename

    # List of variables to save
    if collect_responses:
        variables_to_output = ['all_precisions', 'args', 'num_repetitions', 'output_string', 'power_law_params', 'repet_i', 'all_responses', 'all_targets', 'all_nontargets']
    else:
        variables_to_output = ['all_precisions', 'args', 'num_repetitions', 'output_string', 'power_law_params', 'repet_i']

    print "Doing do_multiple_memory_curve"
    print "max_T: %s" % args.T
    print "File: %s" % output_string

    all_precisions = np.zeros((args.T, args.num_repetitions))
    power_law_params = np.zeros(2)

    if collect_responses:
        all_responses = np.zeros((args.T, args.num_repetitions, args.N))
        all_targets = np.zeros((args.T, args.num_repetitions, args.N))
        all_nontargets = np.zeros((args.T, args.num_repetitions, args.N, args.T-1))

    # Construct different datasets, with t objects
    for repet_i in np.arange(args.num_repetitions):

        for t in np.arange(args.T):

            #### Get multiple examples of precisions, for different number of neurons. #####
            
            if args.code_type == 'conj':
                random_network = RandomFactorialNetwork.create_full_conjunctive(args.M, R=args.R, scale_moments=(args.rc_scale, 0.0001), ratio_moments=(1.0, 0.0001))
            elif args.code_type == 'feat':
                random_network = RandomFactorialNetwork.create_full_features(args.M, R=args.R, scale=args.rc_scale, ratio=40.)
            elif args.code_type == 'mixed':
                conj_params = dict(scale_moments=(args.rc_scale, 0.001), ratio_moments=(1.0, 0.0001))
                feat_params = dict(scale=args.rc_scale2, ratio=40.)
                random_network = RandomFactorialNetwork.create_mixed(args.M, R=args.R, ratio_feature_conjunctive=args.ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
            else:
                raise ValueError('Code_type is wrong!')

            
            # Construct the real dataset
            data_gen = DataGeneratorRFN(args.N, t+1, random_network, sigma_y=args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters, cued_feature_time=t, stimuli_generation=args.stimuli_generation)
            
            # Measure the noise structure
            data_gen_noise = DataGeneratorRFN(3000, t+1, random_network, sigma_y=args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters, cued_feature_time=t, stimuli_generation=args.stimuli_generation)
            stat_meas = StatisticsMeasurer(data_gen_noise)
            # stat_meas = StatisticsMeasurer(data_gen)
            
            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=t)
            
            print "  doing T=%d %d/%d" % (t+1, repet_i+1, args.num_repetitions)

            if args.inference_method == 'sample':
                # Sample thetas
                sampler.sample_theta(num_samples=args.num_samples, burn_samples=100, selection_method=args.selection_method, selection_num_samples=args.num_samples, integrate_tc_out=False, debug=False)
            elif args.inference_method == 'max_lik':
                # Just use the ML value for the theta
                sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
            else:
                raise ValueError('Wrong value for inference_method')

            # Save the precision
            all_precisions[t, repet_i] = sampler.get_precision(remove_chance_level=False, correction_theo_fit=1.0)
            # all_precisions[t, repet_i] = 1./sampler.compute_angle_error()[1]

            print "-> %.5f" % all_precisions[t, repet_i]

            # Collect responses if needed
            if collect_responses:
                (all_responses[t, repet_i], all_targets[t, repet_i], all_nontargets[t, repet_i, :, :t]) = sampler.collect_responses()
            
            # Save to disk, unique filename
            dataio.save_variables(variables_to_output, locals())
        
        xx = np.tile(np.arange(1, args.T+1, dtype='float'), (repet_i+1, 1)).T
        power_law_params = fit_powerlaw(xx, all_precisions[:, :(repet_i+1)], should_plot=True)

        print '====> Power law fits: exponent: %.4f, bias: %.4f' % (power_law_params[0], power_law_params[1])

        # Save to disk, unique filename
        dataio.save_variables(variables_to_output, locals())

    print all_precisions

    
    # Save to disk, unique filename
    dataio.save_variables(variables_to_output, locals())
    

    f = plt.figure()
    ax = f.add_subplot(111)
    plot_mean_std_area(np.arange(args.T), np.mean(all_precisions, 1), np.std(all_precisions, 1), ax_handle=ax)
    ax.set_xlabel('Number of objects')
    ax.set_ylabel('Precision [rad]')
    
    print "Done: %s" % output_string
    return locals()


def launcher_plot_multiple_memory_curve(args):

    input_filename = args.input_filename

    assert input_filename is not '', "Give a file with saved results from do_multiple_memory_curve"

    loaded_data = np.load(input_filename).item()

    all_precisions = loaded_data['all_precisions']
    T = loaded_data['args'].T
    
    f = plt.figure()
    ax = f.add_subplot(111)
    for t in np.arange(T):
        t_space_aligned_right = (T - np.arange(t+1))[::-1]
        # plot_mean_std_area(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], np.std(1./all_precisions[t], 1)[:t+1], ax_handle=ax)
        # semilogy_mean_std_area(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], np.std(1./all_precisions[t], 1)[:t+1], ax_handle=ax)
        # plt.semilogy(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], 'o-', linewidth=2, markersize=8)
        plt.plot(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], 'o-')
    x_labels = ['-%d' % x for x in np.arange(T)[::-1]]
    x_labels[-1] = 'Last'

    ax.set_xticks(t_space_aligned_right)
    ax.set_xticklabels(x_labels)
    ax.set_xlim((0.8, T+0.2))
    ax.set_ylim((1.01, 40))
    # ax.set_xlabel('Recall time')
    # ax.set_ylabel('Precision [rad]')
    legends=['%d items' % (x+1) for x in np.arange(T)]
    legends[0] = '1 item'
    plt.legend(legends, loc='best', numpoints=1, fancybox=True)

    return locals()




def launcher_plot_multiple_memory_curve_simult(args):
    input_filename = args.input_filename

    assert input_filename is not '', "Give a file with saved results from do_multiple_memory_curve"

    loaded_data = np.load(input_filename).item()

    all_precisions = loaded_data['all_precisions']
    T = loaded_data['args'].T
    
    # Average over repetitions, and then get mean across T
    # mean_precision = np.zeros(T)
    # std_precision = np.zeros(T)
    # for t in np.arange(T):
        # mean_precision[t] = np.mean(all_precisions[t][:t+1])
        # std_precision[t] = np.std(all_precisions[t][:t+1])

    mean_precision = np.mean(all_precisions, axis=1)
    std_precision = np.std(all_precisions, axis=1)

    f = plt.figure()
    ax = f.add_subplot(111)
    plt.semilogy(np.arange(1, T+1), mean_precision, 'o-')
    plt.xticks(np.arange(1, T+1))
    plt.xlim((0.9, T+0.1))
    ax.set_xlabel('Number of stored items')
    ax.set_ylabel('Precision [rad]')

    f = plt.figure()
    ax = f.add_subplot(111)
    plt.plot(np.arange(1, T+1), mean_precision)
    ax.set_xlabel('Number of stored items')
    ax.set_ylabel('Precision [rad]')

    plot_mean_std_area(np.arange(1, T+1), mean_precision, std_precision)
    plt.xlabel('Number of stored items')
    plt.ylabel('Precision [rad]^-0.5')


    return locals()
