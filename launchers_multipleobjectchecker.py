#!/usr/bin/env python
# encoding: utf-8
"""
launchers_multipleobjectchecker.py

Created by Loic Matthey on 2012-11-13
Copyright (c) 2012 . All rights reserved.
"""

import matplotlib.pyplot as plt

from datagenerator import *
from randomfactorialnetwork import *
from statisticsmeasurer import *
from slicesampler import *
from utils import *
from dataio import *
from datapbs import *
from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *
import progress


def launcher_do_checks_2obj(args):
    '''
        Perform a series of checks to verify what is happening for the precision using samples for
        2 objects...
    '''

    all_parameters = vars(args)

    dataio = DataIO(output_folder=args.output_directory, label=args.label)
    variables_to_save = ['min_distance_space', 'fi_curve', 'fi_theo_covmeas', 'fi_theo_covtheo', 'fi_precision', 'bias_avg', 'fi_bias']

    save_every = 5
    run_counter = 0

    # num_repetitions = all_parameters['num_repetitions']

    # Force 2 objects...
    # all_parameters['T'] = 2

    # all_parameters['enforce_min_distance'] = 0.1

    # min_distance_space = np.linspace(0.01, 0.8, 10.)
    min_distance_space = np.array([all_parameters['enforce_min_distance']])

    fi_curve = np.nan*np.empty((min_distance_space.size))
    fi_var = np.nan*np.empty((min_distance_space.size))
    fi_theo_covmeas = np.nan*np.empty((min_distance_space.size))
    fi_theo_covtheo = np.nan*np.empty((min_distance_space.size))
    fi_precision = np.nan*np.empty((min_distance_space.size))
    bias_avg = np.nan*np.empty((min_distance_space.size))
    fi_bias = np.nan*np.empty((min_distance_space.size))

    search_progress = progress.Progress(min_distance_space.size)

    for i, min_distance in enumerate(min_distance_space):

        all_parameters['enforce_min_distance'] = min_distance
        print "Min distance: %.2f, %.2f%%, %s left - %s" % (min_distance, search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

        # Initialise everything
        print "init"
        (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)


        # Compute some quantities
        print 'Curve'
        fi_curve[i] = sampler.estimate_fisher_info_from_posterior_avg_randomsubset(subset_size=10)
        print 'Var post'
        fi_var[i] = sampler.estimate_precision_from_posterior_avg_randomsubset(subset_size=10)

        print 'Theo'
        fi_theo_covmeas[i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=False)
        fi_theo_covtheo[i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=True)

        print 'Precision'
        if all_parameters['inference_method'] == 'sample':
            # Sample thetas
            print " sampling..."
            sampler.sample_theta(num_samples=all_parameters['num_samples'], burn_samples=100, selection_method=all_parameters['selection_method'], selection_num_samples=all_parameters['selection_num_samples'], integrate_tc_out=False, debug=False)
        elif all_parameters['inference_method'] == 'max_lik':
            # Just use the ML value for the theta
            # Set to ML value
            print ' setting to ML values...'
            sampler.set_theta_max_likelihood(num_points=100, post_optimise=True)

        fi_precision[i] = sampler.get_precision()

        (angle_stats, angle_errors) = sampler.compute_angle_error(return_errors=True)
        bias_avg[i] = angle_stats['bias']
        fi_bias[i] = 1./np.mean(angle_errors**2.)

        # Compare the FI obtained
        print "Curve: %.2f, Var post: %.2f, Theory meas cov: %.2f, Theory cov theo: %.2f, Precision: %.2f, Bias: %.4g, FI bias: %.4g" % (fi_curve[i], fi_var[i], fi_theo_covmeas[i], fi_theo_covtheo[i], fi_precision[i], bias_avg[i], fi_bias[i])


        search_progress.increment()
        if run_counter % save_every == 0 or search_progress.done():
            dataio.save_variables(variables_to_save, locals())
        run_counter += 1

    # if say_completed:
    #     try:
    #         import sh
    #         sh.say('Work complete')
    #     except Exception:
    #         pass

    if min_distance_space.size>1:
        plt.figure()
        plt.plot(min_distance_space, bias_avg)
        plt.xlabel('Min distance')
        plt.title('Bias as fct of min distance')


    return locals()


def init_everything(parameters):

    # Build the random network
    random_network = init_random_network(parameters)

    # Construct the real dataset
    time_weights_parameters = dict(weighting_alpha=parameters['alpha'], weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')
    cued_feature_time = parameters['T']-1

    # print "Building the database"
    data_gen = DataGeneratorRFN(parameters['N'], parameters['T'], random_network, sigma_y=parameters['sigmay'], sigma_x=parameters['sigmax'], time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time, stimuli_generation=parameters['stimuli_generation'], enforce_min_distance=parameters['enforce_min_distance'])

    # Measure the noise structure
    # print "Measuring noise structure"
    # noise_stimuli_generation = parameters['stimuli_generation']
    # if parameters['stimuli_generation'] == 'random_smallrange':
    #     noise_stimuli_generation = 'random'
    noise_stimuli_generation = 'random'

    data_gen_noise = DataGeneratorRFN(5000, parameters['T'], random_network, sigma_y=parameters['sigmay'], sigma_x=parameters['sigmax'], time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time, stimuli_generation=noise_stimuli_generation, enforce_min_distance=parameters['enforce_min_distance'])
    stat_meas = StatisticsMeasurer(data_gen_noise)

    sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=cued_feature_time)

    return (random_network, data_gen, stat_meas, sampler)


def init_random_network(parameters):

    # Build the random network

    if parameters['code_type'] == 'conj':
        random_network = RandomFactorialNetwork.create_full_conjunctive(parameters['M'], R=parameters['R'], scale_moments=(parameters['rc_scale'], 0.0001), ratio_moments=(1.0, 0.0001))
    elif parameters['code_type'] == 'feat':
        random_network = RandomFactorialNetwork.create_full_features(parameters['M'], R=parameters['R'], scale=parameters['rc_scale'], ratio=40.)
    elif parameters['code_type'] == 'mixed':
        conj_params = dict(scale_moments=(parameters['rc_scale'], 0.001), ratio_moments=(1.0, 0.0001))
        feat_params = dict(scale=parameters['rc_scale2'], ratio=40.)

        random_network = RandomFactorialNetwork.create_mixed(parameters['M'], R=parameters['R'], ratio_feature_conjunctive=ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
    elif code_type == 'wavelet':
        random_network = RandomFactorialNetwork.create_wavelet(parameters['M'], R=parameters['R'], scales_number=5)
    else:
        raise ValueError('Code_type is wrong!')

    return random_network



def launcher_do_separation_rcdependence(args):
    '''
        Look at bias and posterior variance as a function of rc scale,
        for a given M and sigma.
        Work only on a specific set of stimuli (diagonally separated)
    '''
    all_parameters = vars(args)

    dataio = DataIO(output_folder=args.output_directory, label=args.label)
    variables_to_save = ['rcscale_space', 'fi_curve', 'fi_var', 'fi_theo_covmeas', 'fi_theo_covtheo', 'fi_bias']

    save_every = 5
    run_counter = 0

    num_repetitions = all_parameters['num_repetitions']

    # Force 2 objects...
    # all_parameters['T'] = 2

    # all_parameters['enforce_min_distance'] = 0.1
    rcscale_space = np.linspace(0.1, 10., 10.)

    fi_curve = np.nan*np.empty((rcscale_space.size, num_repetitions))
    fi_var = np.nan*np.empty((rcscale_space.size, num_repetitions))
    fi_theo_covmeas = np.nan*np.empty((rcscale_space.size, num_repetitions))
    fi_theo_covtheo = np.nan*np.empty((rcscale_space.size, num_repetitions))
    fi_bias = np.nan*np.empty((rcscale_space.size, num_repetitions))

    search_progress = progress.Progress(rcscale_space.size*num_repetitions)

    for r_i in xrange(num_repetitions):
        for i, rc_scale in enumerate(rcscale_space):

            all_parameters['rc_scale'] = rc_scale

            print "Rcscale: %.2f (%d/%d) %.2f%%, %s left - %s" % (rc_scale, r_i+1, num_repetitions, search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

            # Initialise everything
            print "init"
            (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)

            # Compute some quantities
            print 'Curve'
            fi_curve[i, r_i] = sampler.estimate_fisher_info_from_posterior_avg_randomsubset(subset_size=20)
            print 'Var post'
            fi_var[i, r_i] = sampler.estimate_precision_from_posterior_avg_randomsubset(subset_size=20)

            print 'Theo'
            fi_theo_covmeas[i, r_i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=False)
            fi_theo_covtheo[i, r_i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=True)

            (_, ml_val) = sampler.estimate_truevariance_from_posterior(n=0, return_mean=True)
            fi_bias[i, r_i] = 1./(sampler.data_gen.stimuli_correct[0, sampler.tc[0], sampler.theta_to_sample[0]] - ml_val)**2.

            # Compare the FI obtained
            print "Curve: %.2f, Var post: %.2f, Theory meas cov: %.2f, Theory cov theo: %.2f, FI bias: %.4g" % (fi_curve[i, r_i], fi_var[i, r_i], fi_theo_covmeas[i, r_i], fi_theo_covtheo[i, r_i], fi_bias[i, r_i])


            search_progress.increment()
            if run_counter % save_every == 0 or search_progress.done():
                dataio.save_variables(variables_to_save, locals())
            run_counter += 1

    # if say_completed:
    #     try:
    #         import sh
    #         sh.say('Work complete')
    #     except Exception:
    #         pass

    fi_var_mean = np.mean(fi_var, axis=-1)
    fi_curve_mean = np.mean(fi_curve, axis=-1)

    if rcscale_space.size>1:
        plt.figure()
        plt.plot(rcscale_space, fi_var_mean - fi_curve_mean)
        plt.xlabel('Rc scale')
        plt.title('Variance as fct of rcscale')

        plt.figure()
        plt.plot(rcscale_space[1:], fi_var_mean[1:]/fi_curve_mean[1:])
        plt.plot(rcscale_space, np.ones(rcscale_space.size), 'r--')
        plt.xlabel('Rc scale')
        plt.title('Variance as fct of rcscale')


    return locals()

