#!/usr/bin/env python
# encoding: utf-8
"""
launchers_memorycurves.py


Created by Loic Matthey on 2012-10-10
Copyright (c) 2012 . All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spint

from datagenerator import *
from randomfactorialnetwork import *
from statisticsmeasurer import *
from slicesampler import *
from utils import *
from dataio import *
from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *

from mpl_toolkits.mplot3d import Axes3D


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
    for repet_i in xrange(args.num_repetitions):
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
        
        for t in xrange(args.T):
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
    for t in xrange(args.T):

        for repet_i in xrange(args.num_repetitions):
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
    for t in xrange(args.T):
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
    for repet_i in xrange(args.num_repetitions):

        for t in xrange(args.T):

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
            data_gen_noise = DataGeneratorRFN(5000, t+1, random_network, sigma_y=args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters, cued_feature_time=t, stimuli_generation=args.stimuli_generation)
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
        
        if args.T>1:
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
    for t in xrange(T):
        t_space_aligned_right = (T - np.arange(t+1))[::-1]
        # plot_mean_std_area(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], np.std(1./all_precisions[t], 1)[:t+1], ax_handle=ax)
        # semilogy_mean_std_area(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], np.std(1./all_precisions[t], 1)[:t+1], ax_handle=ax)
        # plt.semilogy(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], 'o-', linewidth=2, markersize=8)
        plt.plot(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], 'o-')
    x_labels = ['-%d' % x for x in xrange(T)[::-1]]
    x_labels[-1] = 'Last'

    ax.set_xticks(t_space_aligned_right)
    ax.set_xticklabels(x_labels)
    ax.set_xlim((0.8, T+0.2))
    ax.set_ylim((1.01, 40))
    # ax.set_xlabel('Recall time')
    # ax.set_ylabel('Precision [rad]')
    legends=['%d items' % (x+1) for x in xrange(T)]
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
    # for t in xrange(T):
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

####################################


def launcher_do_memorycurve_theoretical(args):
    '''
        Compute the FI for different number of items (T)

        Should have a 1/x dependence on T, power law of exponent -1.
    '''

    all_parameters = vars(args)
    data_to_plot = {}
    
    dataio = DataIO(output_folder=args.output_directory, label=args.label)
    variables_to_save = ['rcscale_space', 'sigma_space', 'T_space', 'FI_rc_curv_mult', 'FI_rc_var_mult', 'FI_rc_precision_mult', 'FI_rc_theo_mult', 'repet_i', 'num_repetitions']
    
    save_every = 5
    run_counter = 0
    use_theoretical_cov = all_parameters['use_theoretical_cov']
    print "Use theo cov: %d" % use_theoretical_cov
    
    num_repetitions = all_parameters['num_repetitions']
    check_theoretical_cov = False
    do_curvature = True
    do_precision = False
    do_var = True

    # rcscale_space = np.linspace(0.5, 15.0, 21.)
    rcscale_space = np.linspace(0.01, 15., 21.)
    # rcscale_space = np.linspace(4.0, 4.0, 1.)
    # rcscale_space = np.linspace(all_parameters['rc_scale'], all_parameters['rc_scale'], 1.)
    
    sigma_space = np.linspace(0.01, 0.8, 20.)
    # sigma_space = np.linspace(0.1, 0.1, 1.0)
    # sigma_space = np.linspace(all_parameters['sigmax'], all_parameters['sigmax'], 1.)

    T_space = np.arange(1, all_parameters['T']+1)

    FI_rc_curv_mult = np.zeros((rcscale_space.size, sigma_space.size, T_space.size, 2, num_repetitions), dtype=float)
    FI_rc_var_mult = np.zeros((rcscale_space.size, sigma_space.size, T_space.size, 2, num_repetitions), dtype=float)
    FI_rc_precision_mult = np.zeros((rcscale_space.size, sigma_space.size, T_space.size, num_repetitions), dtype=float)
    FI_rc_theo_mult = np.zeros((rcscale_space.size, sigma_space.size, T_space.size, num_repetitions), dtype=float)
    
    # Show the progress in a nice way
    search_progress = progress.Progress(rcscale_space.size*sigma_space.size*T_space.size*num_repetitions)

    print rcscale_space
    print sigma_space

    for repet_i in xrange(num_repetitions):
        for j, sigma in enumerate(sigma_space):
            for i, rc_scale in enumerate(rcscale_space):
                for t_i, t in enumerate(T_space):

                    ### Estimate the Fisher Information
                    print "FI T effect, T: %d/%d, rcscale %.3f, sigma %.3f (%d/%d). %.2f%%, %s left - %s" % (t, T_space[-1], rc_scale, sigma, repet_i+1, num_repetitions, search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

                    # Current parameter values
                    all_parameters['rc_scale']  = rc_scale
                    all_parameters['sigmax']    = sigma
                    all_parameters['T']         = t

                    ### WORK UNIT

                    # Fisher info
                    ###
                    if use_theoretical_cov:
                        if check_theoretical_cov:
                            (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)

                            computed_cov = random_network.compute_covariance_KL(sigma_2=(all_parameters['sigmax']**2. + all_parameters['sigmay']**2.), T=t, beta=1.0, precision=50)

                            cov_div = np.mean((sampler.noise_covariance-computed_cov)**2.)
                            if cov_div > 0.00001:
                                print cov_div
                                print all_parameters

                                pcolor_2d_data(computed_cov)
                                pcolor_2d_data(sampler.noise_covariance)
                                plt.show()

                                raise ValueError('Big divergence between measured and theoretical divergence!')
                        else:
                            random_network = init_random_network(all_parameters)

                            computed_cov = random_network.compute_covariance_KL(sigma_2=(all_parameters['sigmax']**2. + all_parameters['sigmay']**2.), T=t, beta=1.0, precision=50)


                        # Estimate the fisher information here only
                        print "theoretical FI"
                        FI_rc_theo_mult[i, j, t_i, repet_i] = random_network.compute_fisher_information(stimulus_input=(0.0, 0.0), cov_stim=computed_cov, kappa_different=True)
                        print FI_rc_theo_mult[i, j, t_i, repet_i]

                    else:
                        (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)
                        computed_cov = sampler.noise_covariance
                        # computed_cov = stat_meas.model_parameters['covariances'][-1, 0]

                        # Fisher info
                        print "theoretical FI"
                        FI_rc_theo_mult[i, j, t_i, repet_i] = random_network.compute_fisher_information(stimulus_input=(0.0, 0.0), cov_stim=computed_cov, kappa_different=True)
                        print FI_rc_theo_mult[i, j, t_i, repet_i]

                        # Estimate the rest, possible here.
                        if do_curvature:
                            print "from curvature..."
                            fi_curv_dict = sampler.estimate_fisher_info_from_posterior_avg_randomsubset(subset_size=20, num_points=1000, full_stats=True)
                            (FI_rc_curv_mult[i, j, t_i, 0, repet_i], FI_rc_curv_mult[i, j, t_i, 1, repet_i]) = (fi_curv_dict['mean'], fi_curv_dict['std'])
                            print FI_rc_curv_mult[i, j, t_i, :, repet_i]
                        
                        if do_var:
                                print "from variance of posterior..."
                                fi_var_dict = sampler.estimate_precision_from_posterior_avg_randomsubset(subset_size=20, num_points=1000, full_stats=True)
                                FI_rc_var_mult[i, j, t_i, 0, repet_i], FI_rc_var_mult[i, j, t_i, 1, repet_i] = (fi_var_dict['mean'], fi_var_dict['std'])
                                print FI_rc_var_mult[i, j, t_i, :, repet_i]
                
                        if do_precision:
                            print "from precision of recall..."
                            if all_parameters['inference_method'] == 'sample':
                                # Sample thetas
                                sampler.sample_theta(num_samples=all_parameters['num_samples'], burn_samples=100, selection_method=all_parameters['selection_method'], selection_num_samples=all_parameters['num_samples'], integrate_tc_out=False, debug=False)
                            elif all_parameters['inference_method'] == 'max_lik':
                                # Just use the ML value for the theta
                                sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
                            
                            FI_rc_precision_mult[i, j, t_i, repet_i] = sampler.get_precision()
                            print FI_rc_precision_mult[i, j, t_i, repet_i]

                    ### DONE WORK UNIT

                    search_progress.increment()

                    if run_counter % save_every == 0 or search_progress.done():
                        dataio.save_variables(variables_to_save, locals())

                        # plots
                        for curr_data in variables_to_save:
                            data_to_plot[curr_data] = locals()[curr_data]

                        # plots_fisher_info_param_search(data_to_plot, dataio)

                    run_counter += 1


    return locals()


def launcher_do_memorycurve_theoretical_pbs(args):
    '''
        Compute Fisher info for T objects.

        Get the theoretical FI and the posterior variance estimate as well
    '''

    all_parameters = vars(args)
    
    dataio = DataIO(output_folder=args.output_directory, label=args.label)
    variables_to_save = ['rcscale_space', 'sigma_space', 'M_space', 'T_space', 'FI_rc_theo_mult', 'repet_i', 'num_repetitions', 'use_theoretical_cov']
    
    save_every = 5
    run_counter = 0
    use_theoretical_cov = all_parameters['use_theoretical_cov']
    print "Use theo cov: %d" % use_theoretical_cov
    
    num_repetitions = all_parameters['num_repetitions']
    check_theoretical_cov = False
    do_curvature = False
    do_precision = False
    do_var = True

    rcscale_space = np.linspace(all_parameters['rc_scale'], all_parameters['rc_scale'], 1.)
    sigma_space = np.linspace(all_parameters['sigmax'], all_parameters['sigmax'], 1.)
    M_space = np.array([all_parameters['M']])
    T_space = np.arange(1, all_parameters['T']+1)
    
    FI_rc_theo_mult = np.zeros((rcscale_space.size, sigma_space.size, M_space.size, T_space.size, num_repetitions), dtype=float)
    
    if do_curvature:
        variables_to_save.append('FI_rc_curv_mult')
        FI_rc_curv_mult = np.zeros((rcscale_space.size, sigma_space.size, M_space.size, T_space.size, 2, num_repetitions), dtype=float)
    if do_precision:
        variables_to_save.append('FI_rc_precision_mult')
        FI_rc_precision_mult = np.zeros((rcscale_space.size, sigma_space.size, M_space.size, T_space.size, num_repetitions), dtype=float)
    if do_var:
        variables_to_save.append('FI_rc_var_mult')
        FI_rc_var_mult = np.zeros((rcscale_space.size, sigma_space.size, M_space.size, T_space.size, 2, num_repetitions), dtype=float)

    # Show the progress in a nice way
    search_progress = progress.Progress(rcscale_space.size*sigma_space.size*T_space.size*M_space.size*num_repetitions)

    print rcscale_space
    print sigma_space
    print M_space
    print T_space

    for repet_i in xrange(num_repetitions):
        for j, sigma in enumerate(sigma_space):
            for i, rc_scale in enumerate(rcscale_space):
                for m_i, M in enumerate(M_space):
                    for t_i, t in enumerate(T_space):
                        ### Estimate the Fisher Information
                        print "FI T effect, T: %d/%d, rcscale %.3f, sigma %.3f, M %d, (%d/%d). %.2f%%, %s left - %s" % (t, T_space[-1], rc_scale, sigma, M, repet_i+1, num_repetitions, search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

                        # Current parameter values
                        all_parameters['rc_scale']  = rc_scale
                        all_parameters['sigmax']    = sigma
                        all_parameters['M']         = M
                        all_parameters['T']         = t

                        ### WORK UNIT

                        # Fisher info
                        ###
                        if use_theoretical_cov:
                            if check_theoretical_cov:
                                (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)

                                computed_cov = random_network.compute_covariance_KL(sigma_2=(all_parameters['sigmax']**2. + all_parameters['sigmay']**2.), T=t, beta=1.0, precision=50)

                                cov_div = np.mean((sampler.noise_covariance-computed_cov)**2.)
                                if cov_div > 0.00001:
                                    print cov_div
                                    print all_parameters

                                    pcolor_2d_data(computed_cov)
                                    pcolor_2d_data(sampler.noise_covariance)
                                    plt.show()

                                    raise ValueError('Big divergence between measured and theoretical divergence!')
                            else:
                                random_network = init_random_network(all_parameters)

                                computed_cov = random_network.compute_covariance_KL(sigma_2=(all_parameters['sigmax']**2. + all_parameters['sigmay']**2.), T=t, beta=1.0, precision=50)


                            # Estimate the fisher information here only
                            print "theoretical FI"
                            FI_rc_theo_mult[i, j, m_i, t_i, repet_i] = random_network.compute_fisher_information(stimulus_input=(0.0, 0.0), cov_stim=computed_cov, kappa_different=True)
                            print FI_rc_theo_mult[i, j, m_i, t_i, repet_i]

                        else:
                            (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)
                            computed_cov = sampler.noise_covariance
                            # computed_cov = stat_meas.model_parameters['covariances'][-1, 0]

                            # Fisher info
                            print "theoretical FI"
                            FI_rc_theo_mult[i, j, m_i, t_i, repet_i] = random_network.compute_fisher_information(stimulus_input=(0.0, 0.0), cov_stim=computed_cov, kappa_different=True)
                            print FI_rc_theo_mult[i, j, m_i, t_i, repet_i]

                            # Estimate the rest, possible here.
                            if do_curvature:
                                print "from curvature..."
                                fi_curv_dict = sampler.estimate_fisher_info_from_posterior_avg_randomsubset(subset_size=20, num_points=1000, full_stats=True)
                                (FI_rc_curv_mult[i, j, m_i, t_i, 0, repet_i], FI_rc_curv_mult[i, j, m_i, t_i, 1, repet_i]) = (fi_curv_dict['mean'], fi_curv_dict['std'])
                                print FI_rc_curv_mult[i, j, m_i, t_i, :, repet_i]
                            
                            if do_var:
                                print "from variance of posterior..."
                                fi_var_dict = sampler.estimate_precision_from_posterior_avg_randomsubset(subset_size=20, num_points=1000, full_stats=True)
                                FI_rc_var_mult[i, j, m_i, t_i, 0, repet_i], FI_rc_var_mult[i, j, m_i, t_i, 1, repet_i] = (fi_var_dict['mean'], fi_var_dict['std'])
                                print FI_rc_var_mult[i, j, m_i, t_i, :, repet_i]
                    
                            if do_precision:
                                print "from precision of recall..."
                                if all_parameters['inference_method'] == 'sample':
                                    # Sample thetas
                                    sampler.sample_theta(num_samples=all_parameters['num_samples'], burn_samples=100, selection_method=all_parameters['selection_method'], selection_num_samples=all_parameters['num_samples'], integrate_tc_out=False, debug=False)
                                elif all_parameters['inference_method'] == 'max_lik':
                                    # Just use the ML value for the theta
                                    sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
                                
                                FI_rc_precision_mult[i, j, m_i, t_i, repet_i] = sampler.get_precision()
                                print FI_rc_precision_mult[i, j, m_i, t_i, repet_i]

                        ### DONE WORK UNIT

                        search_progress.increment()

                        if run_counter % save_every == 0 or search_progress.done():
                            dataio.save_variables(variables_to_save, locals())

                        run_counter += 1
    return locals()


def launcher_do_memorycurve_theoretical_pbs_theoonly(args):
    '''
        Compute Fisher info for T objects.

        Get the theoretical FI and the posterior variance estimate as well
    '''

    all_parameters = vars(args)
    
    dataio = DataIO(output_folder=args.output_directory, label=args.label)
    variables_to_save = ['rcscale_space', 'sigma_space', 'M_space', 'T_space', 'FI_rc_theo_mult', 'repet_i', 'num_repetitions', 'use_theoretical_cov']
    
    save_every = 5
    run_counter = 0
    use_theoretical_cov = True
    print "Use theo cov: %d" % use_theoretical_cov
    
    num_repetitions = all_parameters['num_repetitions']
    check_theoretical_cov = False
    do_curvature = False
    do_precision = False
    do_var = False

    rcscale_space = np.linspace(all_parameters['rc_scale'], all_parameters['rc_scale'], 1.)
    sigma_space = np.linspace(all_parameters['sigmax'], all_parameters['sigmax'], 1.)
    T_space = np.arange(1, all_parameters['T']+1)
    
    # M_space = np.array([all_parameters['M']])
    # M_space = np.arange(5, 30, 3, dtype=int)**2.  # Ease the load on PBS...
    # M_space = np.floor(np.linspace(25, 900, 49)).astype(int)
    M_space = np.arange(5, 22, dtype=int)**2.
    
    FI_rc_theo_mult = np.zeros((rcscale_space.size, sigma_space.size, M_space.size, T_space.size, num_repetitions), dtype=float)
    
    if do_curvature:
        variables_to_save.append('FI_rc_curv_mult')
        FI_rc_curv_mult = np.zeros((rcscale_space.size, sigma_space.size, M_space.size, T_space.size, 2, num_repetitions), dtype=float)
    if do_precision:
        variables_to_save.append('FI_rc_precision_mult')
        FI_rc_precision_mult = np.zeros((rcscale_space.size, sigma_space.size, M_space.size, T_space.size, num_repetitions), dtype=float)
    if do_var:
        variables_to_save.append('FI_rc_var_mult')
        FI_rc_var_mult = np.zeros((rcscale_space.size, sigma_space.size, M_space.size, T_space.size, 2, num_repetitions), dtype=float)

    # Show the progress in a nice way
    search_progress = progress.Progress(rcscale_space.size*sigma_space.size*T_space.size*M_space.size*num_repetitions)

    print rcscale_space
    print sigma_space
    print M_space
    print T_space

    for repet_i in xrange(num_repetitions):
        for j, sigma in enumerate(sigma_space):
            for i, rc_scale in enumerate(rcscale_space):
                for m_i, M in enumerate(M_space):
                    for t_i, t in enumerate(T_space):
                        ### Estimate the Fisher Information
                        print "FI T effect, T: %d/%d, rcscale %.3f, sigma %.3f, M %d, (%d/%d). %.2f%%, %s left - %s" % (t, T_space[-1], rc_scale, sigma, M, repet_i+1, num_repetitions, search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

                        # Current parameter values
                        all_parameters['rc_scale']  = rc_scale
                        all_parameters['sigmax']    = sigma
                        all_parameters['M']         = M
                        all_parameters['T']         = t

                        ### WORK UNIT

                        # Fisher info
                        ###
                        if use_theoretical_cov:
                            if check_theoretical_cov:
                                (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)

                                computed_cov = random_network.compute_covariance_KL(sigma_2=(all_parameters['sigmax']**2. + all_parameters['sigmay']**2.), T=t, beta=1.0, precision=50)

                                cov_div = np.mean((sampler.noise_covariance-computed_cov)**2.)
                                if cov_div > 0.00001:
                                    print cov_div
                                    print all_parameters

                                    pcolor_2d_data(computed_cov)
                                    pcolor_2d_data(sampler.noise_covariance)
                                    plt.show()

                                    raise ValueError('Big divergence between measured and theoretical divergence!')
                            else:
                                random_network = init_random_network(all_parameters)

                                computed_cov = random_network.compute_covariance_KL(sigma_2=(all_parameters['sigmax']**2. + all_parameters['sigmay']**2.), T=t, beta=1.0, precision=50)


                            # Estimate the fisher information here only
                            print "theoretical FI"
                            FI_rc_theo_mult[i, j, m_i, t_i, repet_i] = random_network.compute_fisher_information(stimulus_input=(0.0, 0.0), cov_stim=computed_cov)
                            print FI_rc_theo_mult[i, j, m_i, t_i, repet_i]

                        else:
                            (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)
                            computed_cov = sampler.noise_covariance
                            # computed_cov = stat_meas.model_parameters['covariances'][-1, 0]

                            # Fisher info
                            print "theoretical FI"
                            FI_rc_theo_mult[i, j, m_i, t_i, repet_i] = random_network.compute_fisher_information(stimulus_input=(0.0, 0.0), cov_stim=computed_cov)
                            print FI_rc_theo_mult[i, j, m_i, t_i, repet_i]

                            # Estimate the rest, possible here.
                            if do_curvature:
                                print "from curvature..."
                                fi_curv_dict = sampler.estimate_fisher_info_from_posterior_avg_randomsubset(subset_size=20, num_points=1000, full_stats=True)
                                (FI_rc_curv_mult[i, j, m_i, t_i, 0, repet_i], FI_rc_curv_mult[i, j, m_i, t_i, 1, repet_i]) = (fi_curv_dict['mean'], fi_curv_dict['std'])
                                print FI_rc_curv_mult[i, j, m_i, t_i, :, repet_i]
                            
                            if do_var:
                                print "from variance of posterior..."
                                fi_var_dict = sampler.estimate_precision_from_posterior_avg_randomsubset(subset_size=20, num_points=1000, full_stats=True)
                                FI_rc_var_mult[i, j, m_i, t_i, 0, repet_i], FI_rc_var_mult[i, j, m_i, t_i, 1, repet_i] = (fi_var_dict['mean'], fi_var_dict['std'])
                                print FI_rc_var_mult[i, j, m_i, t_i, :, repet_i]
                    
                            if do_precision:
                                print "from precision of recall..."
                                if all_parameters['inference_method'] == 'sample':
                                    # Sample thetas
                                    sampler.sample_theta(num_samples=all_parameters['num_samples'], burn_samples=100, selection_method=all_parameters['selection_method'], selection_num_samples=all_parameters['num_samples'], integrate_tc_out=False, debug=False)
                                elif all_parameters['inference_method'] == 'max_lik':
                                    # Just use the ML value for the theta
                                    sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
                                
                                FI_rc_precision_mult[i, j, m_i, t_i, repet_i] = sampler.get_precision()
                                print FI_rc_precision_mult[i, j, m_i, t_i, repet_i]

                        ### DONE WORK UNIT

                        search_progress.increment()

                        if run_counter % save_every == 0 or search_progress.done():
                            dataio.save_variables(variables_to_save, locals())

                        run_counter += 1
    return locals()

def init_everything(parameters):

    # Build the random network
    random_network = init_random_network(parameters)
        
    # Construct the real dataset
    time_weights_parameters = dict(weighting_alpha=parameters['alpha'], weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')
    cued_feature_time = parameters['T']-1

    # print "Building the database"
    data_gen = DataGeneratorRFN(parameters['N'], parameters['T'], random_network, sigma_y=parameters['sigmay'], sigma_x=parameters['sigmax'], time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time, stimuli_generation=parameters['stimuli_generation'])
    
    # Measure the noise structure
    # print "Measuring noise structure"
    data_gen_noise = DataGeneratorRFN(5000, parameters['T'], random_network, sigma_y=parameters['sigmay'], sigma_x=parameters['sigmax'], time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time, stimuli_generation=parameters['stimuli_generation'])
    stat_meas = StatisticsMeasurer(data_gen_noise)
    
    sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=cued_feature_time)

    return (random_network, data_gen, stat_meas, sampler)



def init_random_network(parameters):

    # Build the random network
    
    if parameters['code_type'] == 'conj':
        random_network = RandomFactorialNetwork.create_full_conjunctive(parameters['M'], R=parameters['R'], scale_moments=(parameters['rc_scale'], 0.0001), ratio_moments=(1.0, 0.0001))
    elif parameters['code_type'] == 'feat':
        random_network = RandomFactorialNetwork.create_full_features(parameters['M'], R=parameters['R'], scale=parameters['rc_scale'], ratio=parameters['feat_ratio'])
    elif parameters['code_type'] == 'mixed':
        conj_params = dict(scale_moments=[parameters['rc_scale'], 0.0001], ratio_moments=[1.0, 0.0001])
        feat_params = dict(scale=parameters['rc_scale2'], ratio=parameters['feat_ratio'])

        random_network = RandomFactorialNetwork.create_mixed(parameters['M'], R=parameters['R'], ratio_feature_conjunctive=parameters['ratio_conj'], conjunctive_parameters=conj_params, feature_parameters=feat_params)
    elif code_type == 'wavelet':
        random_network = RandomFactorialNetwork.create_wavelet(parameters['M'], R=parameters['R'], scales_number=5)
    else:
        raise ValueError('Code_type is wrong!')

    return random_network


def launcher_reload_memorycurve_theoretical(args):
    '''
        Reload results from launcher_do_memorycurve_theoretical and do some plots
    '''

    # Check that a filename was provided
    input_filename = args.input_filename
    assert input_filename is not '', "Give a file with saved results from launcher_do_fisher_information_param_search"


    dataio = DataIO(output_folder=args.output_directory, label=args.label)

    # Reload everything
    loaded_data = np.load(input_filename).item()

    # Plots
    plots_memorycurve_theoretical(loaded_data, dataio, save_figures=False)

    return locals()
    

def plots_memorycurve_theoretical(data_to_plot, dataio=None, save_figures=False):

    # interpolation = 'nearest'
    interpolation = 'bicubic'


    # Sanity check, verify that we have all the data we will be plotting
    required_variables = ['rcscale_space', 'sigma_space', 'FI_rc_theo_mult']

    assert set(required_variables) <= set(data_to_plot), "This dataset is not complete. Require %s" % required_variables

    # The theoretical results
    FI_rc_theo_mean = np.mean(data_to_plot['FI_rc_theo_mult'], axis=-1)

    exp_target = np.array([ 8.81007762,  4.7976755,   3.61554792,  2.4624979,   1.78252039])*2.

    # Check the fits
    dist_expo = np.sum((FI_rc_theo_mean - exp_target)**2., axis=-1)

    pcolor_2d_data(np.log(dist_expo), x=data_to_plot['rcscale_space'], y=data_to_plot['sigma_space'], title='Log distance to experimental memory curve', label_format="%.2f", interpolation=interpolation)

    # Find the best fit
    best_fit_indices = argmin_indices(dist_expo)

    plt.figure()
    plt.plot(data_to_plot['T_space'], FI_rc_theo_mean[tuple(best_fit_indices)], 'o-')
    plt.plot(data_to_plot['T_space'], exp_target, 'o-')
    plt.title('Best fitting theoretical curve: rcscale %.2f, sigma %.2f' % (data_to_plot['rcscale_space'][best_fit_indices[0]], data_to_plot['sigma_space'][best_fit_indices[1]]))
    plt.legend(['Fit', 'Experimental'])
    plt.xlabel('Number of items')
    plt.ylabel('FI')



def launcher_do_memorycurve_theoretical_3d_volume(args):
    '''
        Compute the FI for different number of items (T), over a full 3D volume of parameters

        Should have a 1/x dependence on T, power law of exponent -1.
    '''

    all_parameters = vars(args)
    data_to_plot = {}
    
    dataio = DataIO(output_folder=args.output_directory, label=args.label)
    variables_to_save = ['rcscale_space', 'sigma_space', 'M_space', 'T_space', 'FI_rc_curv_mult', 'FI_rc_precision_mult', 'FI_rc_theo_mult', 'FI_rc_var_mult', 'repet_i', 'num_repetitions']
    
    save_every = 5
    run_counter = 0
    use_theoretical_cov = all_parameters['use_theoretical_cov']
    print "Use theo cov: %d" % use_theoretical_cov
    
    num_repetitions = all_parameters['num_repetitions']
    check_theoretical_cov = False
    do_curvature = False
    do_precision = False
    do_var = False

    # rcscale_space = np.linspace(0.5, 15.0, 21.)
    rcscale_space = np.linspace(0.01, 10., 51.)
    # rcscale_space = np.linspace(4.0, 4.0, 1.)
    # rcscale_space = np.linspace(all_parameters['rc_scale'], all_parameters['rc_scale'], 1.)
    
    sigma_space = np.linspace(0.01, 0.8, 50.)
    # sigma_space = np.linspace(0.1, 0.1, 1.0)
    # sigma_space = np.linspace(all_parameters['sigmax'], all_parameters['sigmax'], 1.)

    # M_space = np.arange(10, 30, 5)**2.
    M_space = np.arange(5, 30, 1)**2.

    T_space = np.arange(1, all_parameters['T']+1)

    FI_rc_curv_mult = np.zeros((rcscale_space.size, sigma_space.size, M_space.size, T_space.size, 2, num_repetitions), dtype=float)
    FI_rc_var_mult = np.zeros((rcscale_space.size, sigma_space.size, M_space.size, T_space.size, 2, num_repetitions), dtype=float)
    FI_rc_precision_mult = np.zeros((rcscale_space.size, sigma_space.size, M_space.size, T_space.size, num_repetitions), dtype=float)
    FI_rc_theo_mult = np.zeros((rcscale_space.size, sigma_space.size, M_space.size, T_space.size, num_repetitions), dtype=float)
    
    # Show the progress in a nice way
    search_progress = progress.Progress(rcscale_space.size*sigma_space.size*M_space.size*T_space.size*num_repetitions)

    print rcscale_space
    print sigma_space
    print M_space

    for repet_i in xrange(num_repetitions):
        for j, sigma in enumerate(sigma_space):
            for i, rc_scale in enumerate(rcscale_space):
                for m_i, M in enumerate(M_space):
                    for t_i, t in enumerate(T_space):

                        ### Estimate the Fisher Information
                        print "FI T effect, T: %d/%d, rcscale %.3f, sigma %.3f, M %d, (%d/%d). %.2f%%, %s left - %s" % (t, T_space[-1], rc_scale, sigma, M, repet_i+1, num_repetitions, search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

                        # Current parameter values
                        all_parameters['rc_scale']  = rc_scale
                        all_parameters['sigmax']    = sigma
                        all_parameters['M']         = M
                        all_parameters['T']         = t

                        ### WORK UNIT

                        # Fisher info
                        ###
                        if use_theoretical_cov:
                            if check_theoretical_cov:
                                (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)

                                computed_cov = random_network.compute_covariance_KL(sigma_2=(all_parameters['sigmax']**2. + all_parameters['sigmay']**2.), T=t, beta=1.0, precision=50)

                                cov_div = np.mean((sampler.noise_covariance-computed_cov)**2.)
                                if cov_div > 0.00001:
                                    print cov_div
                                    print all_parameters

                                    pcolor_2d_data(computed_cov)
                                    pcolor_2d_data(sampler.noise_covariance)
                                    plt.show()

                                    raise ValueError('Big divergence between measured and theoretical divergence!')
                            else:
                                random_network = init_random_network(all_parameters)

                                computed_cov = random_network.compute_covariance_KL(sigma_2=(all_parameters['sigmax']**2. + all_parameters['sigmay']**2.), T=t, beta=1.0, precision=50)


                            # Estimate the fisher information here only
                            print "theoretical FI"
                            FI_rc_theo_mult[i, j, m_i, t_i, repet_i] = random_network.compute_fisher_information(stimulus_input=(0.0, 0.0), cov_stim=computed_cov)
                            print FI_rc_theo_mult[i, j, m_i, t_i, repet_i]

                        else:
                            (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)
                            computed_cov = sampler.noise_covariance
                            # computed_cov = stat_meas.model_parameters['covariances'][-1, 0]

                            # Fisher info
                            print "theoretical FI"
                            FI_rc_theo_mult[i, j, m_i, t_i, repet_i] = random_network.compute_fisher_information(stimulus_input=(0.0, 0.0), cov_stim=computed_cov)
                            print FI_rc_theo_mult[i, j, m_i, t_i, repet_i]

                            # Estimate the rest, possible here.
                            if do_curvature:
                                print "from curvature..."
                                fi_curv_dict = sampler.estimate_fisher_info_from_posterior_avg_randomsubset(subset_size=20, num_points=1000, full_stats=True)
                                (FI_rc_curv_mult[i, j, m_i, t_i, 0, repet_i], FI_rc_curv_mult[i, j, m_i, t_i, 1, repet_i]) = (fi_curv_dict['mean'], fi_curv_dict['std'])
                                print FI_rc_curv_mult[i, j, m_i, t_i, :, repet_i]
                            
                            if do_var:
                                print "from variance of posterior..."
                                fi_var_dict = sampler.estimate_precision_from_posterior_avg_randomsubset(subset_size=20, num_points=1000, full_stats=True)
                                FI_rc_var_mult[i, j, m_i, t_i, 0, repet_i], FI_rc_var_mult[i, j, m_i, t_i, 1, repet_i] = (fi_var_dict['mean'], fi_var_dict['std'])
                                print FI_rc_var_mult[i, j, m_i, t_i, :, repet_i]
                

                            if do_precision:
                                print "from precision of recall..."
                                if all_parameters['inference_method'] == 'sample':
                                    # Sample thetas
                                    sampler.sample_theta(num_samples=all_parameters['num_samples'], burn_samples=100, selection_method=all_parameters['selection_method'], selection_num_samples=all_parameters['num_samples'], integrate_tc_out=False, debug=False)
                                elif all_parameters['inference_method'] == 'max_lik':
                                    # Just use the ML value for the theta
                                    sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
                                
                                FI_rc_precision_mult[i, j, m_i, t_i, repet_i] = sampler.get_precision()
                                print FI_rc_precision_mult[i, j, m_i, t_i, repet_i]

                        ### DONE WORK UNIT

                        search_progress.increment()

                        if run_counter % save_every == 0 or search_progress.done():
                            dataio.save_variables(variables_to_save, locals())

                            # plots
                            for curr_data in variables_to_save:
                                data_to_plot[curr_data] = locals()[curr_data]

                            # plots_fisher_info_param_search(data_to_plot, dataio)

                        run_counter += 1


    return locals()


def launcher_reload_memorycurve_theoretical_3d_volume(args):
    '''
        Reload results from launcher_do_memorycurve_theoretical_3d_volume and do some plots
    '''

    # Check that a filename was provided
    input_filename = args.input_filename
    assert input_filename is not '', "Give a file with saved results from launcher_do_fisher_information_param_search"


    dataio = DataIO(output_folder=args.output_directory, label=args.label)

    # Reload everything
    loaded_data = np.load(input_filename).item()

    # Plots
    plots_outs = plots_memorycurve_theoretical_3dvolume(loaded_data, dataio, save_figures=False)

    return locals()


def plots_memorycurve_theoretical_3dvolume(data_to_plot, dataio=None, save_figures=False):
    '''
        Do plots in 2D and 3D to show the memory fits
    '''

    interpolation = 'nearest'

    FI_rc_theo_mean = np.mean(data_to_plot['FI_rc_theo_mult'], axis=-1)

    exp_target = np.array([ 8.81007762,  4.7976755,   3.61554792,  2.4624979,   1.78252039])*2.

    # Check the fits
    dist_exp = np.sum((FI_rc_theo_mean - exp_target)**2., axis=-1)
    log_dist_exp = np.log(dist_exp)

    dist_1obj = (FI_rc_theo_mean[..., 0] - exp_target[0])**2.

    # xxx = np.array(cross(data_to_plot['rcscale_space'], data_to_plot['sigma_space'], data_to_plot['M_space']))

    # Filter fits too bad
    max_mse = 100.
    filter_indices = np.ravel_multi_index(np.nonzero(dist_exp < max_mse), dist_exp.shape)

    filtered_dist_exp = dist_exp.copy()
    filtered_dist_exp[filtered_dist_exp > max_mse] = np.nan
    
    # Construct smooth region
    # log_dist_exp[log_dist_exp < 3.0] = 2.0

    draw_colorbar = False
    use_mayavi = True
    do_3d = True
    do_pcolor = True

    if do_pcolor:
        for m_i, M in enumerate(data_to_plot['M_space']):
            # Do one pcolor per M
            pcolor_2d_data(log_dist_exp[..., m_i], x=data_to_plot['rcscale_space'], y=data_to_plot['sigma_space'], title='Log distance to experimental memory curve. M %d' % M, label_format="%.2f", interpolation=interpolation)

            # pcolor_2d_data(log_dist_exp[..., m_i], x=data_to_plot['rcscale_space'], y=data_to_plot['sigma_space'], title='Log distance to 1 obj experimental FI. M %d' % M, label_format="%.2f", interpolation=interpolation)


    plt.show()

    if do_3d:
        if use_mayavi:
            # from mayavi import mlab
            # mlab.options.backend = 'envisage'
            import mayavi.mlab as mplt
            
            # mplt.points3d(xxx[filter_indices, 0], xxx[filter_indices, 1], xxx[filter_indices, 2], np.ravel(np.log(dist_exp))[filter_indices])
            # mplt.points3d(xxx[:, 0], xxx[:, 1], xxx[:, 2], scale_factor=0.5)
            # mplt.mesh(x, y, z, scalars=Z_norm, vmin=0.0)

            # mplt.contour3d(np.log(dist_exp), contours=10, transparent=True, extent=[data_to_plot['rcscale_space'].min(), data_to_plot['rcscale_space'].max(), data_to_plot['sigma_space'].min(), data_to_plot['sigma_space'].max(), data_to_plot['M_space'].min(), data_to_plot['M_space'].max()])

            # Interpolate
            if False:
                pts_interpol = 100j
                X, Y, Z = np.mgrid[data_to_plot['rcscale_space'].min():data_to_plot['rcscale_space'].max():pts_interpol, data_to_plot['sigma_space'].min():data_to_plot['sigma_space'].max():pts_interpol, data_to_plot['M_space'].min():data_to_plot['M_space'].max():pts_interpol]

                dist_exp_interp = spint.griddata(xxx, np.ravel(dist_exp), (X, Y, Z))

                mplt.figure()
                mplt.contour3d(log_dist_exp_interp, contours=10, transparent=True)

            # mplt.figure(bgcolor=(0.7,0.7,0.7))
            mplt.figure(bgcolor=(1.0, 1.0, 1.0))
            # mplt.figure()
            mplt.contour3d(log_dist_exp, contours=10, transparent=True)
            mplt.axes(color=(0.0, 0.0, 0.0))
            
            if draw_colorbar:
                mplt.colorbar(title='', orientation='vertical', label_fmt='%.2f', nb_labels=5)
            
            
            # mplt.outline(color=(0., 0., 0.))
            mplt.show()

        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            ax.scatter(xxx[filter_indices, 0], xxx[filter_indices, 1], xxx[filter_indices, 2], c=np.ravel(np.log(dist_exp))[filter_indices])

            ax.set_xlabel('Rc scale')
            ax.set_ylabel('Sigma')
            ax.set_zlabel('M')

            plt.show()

    
    





